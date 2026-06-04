#!/usr/bin/env python3
"""Batch-level normal/attacker detection with a DATE-style detector."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_: object):
        return x

from date_data import (
    DateCollator,
    QueryDataset,
    default_attacker_path,
    default_dataset_paths,
    generate_mask_patterns,
    iter_batches,
    read_jsonl_queries,
)
from date_model import DateDiscriminator


@dataclass
class BatchScore:
    split: str
    batch_id: int
    label: int
    size: int
    attacker_ratio: float
    attacker_count: int
    score: float
    p_value: float
    suspicious: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect attacker query batches using a DATE-style self-supervised text anomaly detector."
    )
    parser.add_argument("--dataset", default="model_leeching")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--normal-source", choices=["dataset", "global"], default="dataset")
    parser.add_argument("--global-normal-path", type=Path, default=Path("data/global_normal/queries.jsonl"))
    parser.add_argument("--normal-path", type=Path, default=None)
    parser.add_argument("--attacker-path", type=Path, default=None)
    parser.add_argument("--text-field", default="query")
    parser.add_argument("--max-normal", type=int, default=None)
    parser.add_argument("--max-attacker", type=int, default=None)
    parser.add_argument("--normal-train-ratio", type=float, default=0.8)

    parser.add_argument("--model-name", default="google/electra-small-discriminator")
    parser.add_argument(
        "--generator-model-name",
        default=None,
        help="MLM generator checkpoint. Defaults to --model-name when --generator learned.",
    )
    parser.add_argument("--generator", choices=["learned", "random"], default="learned")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mask-patterns", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=5000,
        help="Stop DATE training after this many optimizer steps. Use 0 to disable the cap.",
    )
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--rtd-weight", type=float, default=50.0)
    parser.add_argument("--rmd-weight", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto when omitted")

    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument(
        "--test-sidedness",
        choices=["upper", "lower", "two-sided"],
        default="upper",
        help="Tail test for DATE scores. two-sided uses alpha split across both tails.",
    )
    parser.add_argument("--null-samples", type=int, default=1000)
    parser.add_argument("--batch-score-agg", choices=["mean", "median", "max", "quantile", "topk_mean"], default="mean")
    parser.add_argument("--score-quantile", type=float, default=0.9)
    parser.add_argument("--topk-fraction", type=float, default=0.2)
    parser.add_argument("--mixed-attacker-ratios", default="0.05,0.1,0.25,0.5")
    parser.add_argument("--mixed-batches-per-ratio", type=int, default=50)
    parser.add_argument("--benign-eval-batches", type=int, default=0)
    parser.add_argument("--attacker-eval-batches", type=int, default=0)
    parser.add_argument("--keep-last-batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("DATE/outputs"))
    parser.add_argument("--save-model-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.normal_path is None and args.attacker_path is None and args.normal_source == "dataset":
        return default_dataset_paths(args.data_root, args.dataset)
    if args.attacker_path is not None:
        attacker_path = args.attacker_path
        resolved_dataset = args.dataset
    else:
        attacker_path, resolved_dataset = default_attacker_path(args.data_root, args.dataset)
    if args.normal_path is not None:
        normal_path = args.normal_path
    elif args.normal_source == "global":
        normal_path = args.global_normal_path
    else:
        normal_path = args.data_root / resolved_dataset / "normal" / "queries.jsonl"
    if not normal_path.exists():
        raise FileNotFoundError(f"Missing normal queries: {normal_path}")
    return normal_path, attacker_path, resolved_dataset


def random_replace(
    masked_input_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    vocab_size: int,
    forbidden_ids: set[int],
) -> torch.Tensor:
    corrupted = masked_input_ids.clone()
    count = int(mask_positions.sum().item())
    if count == 0:
        return corrupted
    replacement = torch.randint(0, vocab_size, (count,), device=masked_input_ids.device)
    if forbidden_ids:
        forbidden = torch.as_tensor(sorted(forbidden_ids), device=masked_input_ids.device)
        for _ in range(8):
            bad = (replacement[:, None] == forbidden[None, :]).any(dim=1)
            if not bool(bad.any()):
                break
            replacement[bad] = torch.randint(0, vocab_size, (int(bad.sum().item()),), device=masked_input_ids.device)
    corrupted[mask_positions] = replacement
    return corrupted


def learned_replace(generator, masked_input_ids, attention_mask, mask_positions, mlm_labels):
    outputs = generator(input_ids=masked_input_ids, attention_mask=attention_mask, labels=mlm_labels)
    logits = outputs.logits
    sampled = torch.argmax(logits, dim=-1)
    corrupted = masked_input_ids.clone()
    corrupted[mask_positions] = sampled[mask_positions]
    return corrupted, outputs.loss


def train_date(args, tokenizer, discriminator, generator, collator, train_texts, device):
    discriminator.train()
    if generator is not None:
        generator.train()

    params = list(discriminator.parameters())
    if generator is not None:
        params += list(generator.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        QueryDataset(train_texts),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    requested_steps = len(loader) * args.epochs
    total_steps = max(1, min(requested_steps, args.max_train_steps) if args.max_train_steps > 0 else requested_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    forbidden_ids = {
        item for item in [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id,
        ] if item is not None
    }

    history = []
    global_step = 0
    stop_training = False
    for epoch in range(args.epochs):
        totals = {"loss": 0.0, "rtd": 0.0, "rmd": 0.0, "mlm": 0.0}
        seen = 0
        remaining_steps = total_steps - global_step
        epoch_step_limit = min(len(loader), remaining_steps)
        iterator = tqdm(loader, total=epoch_step_limit, desc=f"DATE train epoch {epoch + 1}/{args.epochs}")
        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            mask_positions = batch["mask_positions"]
            if generator is None:
                corrupted = random_replace(
                    batch["masked_input_ids"],
                    mask_positions,
                    vocab_size=len(tokenizer),
                    forbidden_ids=forbidden_ids,
                )
                mlm_loss = torch.zeros((), device=device)
            else:
                corrupted, mlm_loss = learned_replace(
                    generator,
                    batch["masked_input_ids"],
                    attention_mask,
                    mask_positions,
                    batch["mlm_labels"],
                )

            rtd_labels = (corrupted != input_ids).long()
            rtd_labels = rtd_labels.masked_fill(attention_mask == 0, -100)
            rtd_logits, rmd_logits = discriminator(corrupted, attention_mask)
            rtd_loss = F.cross_entropy(rtd_logits.view(-1, 2), rtd_labels.view(-1), ignore_index=-100)
            rmd_loss = F.cross_entropy(rmd_logits, batch["pattern_ids"])
            loss = mlm_loss + args.rtd_weight * rtd_loss + args.rmd_weight * rmd_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            bsz = input_ids.size(0)
            seen += bsz
            totals["loss"] += float(loss.item()) * bsz
            totals["rtd"] += float(rtd_loss.item()) * bsz
            totals["rmd"] += float(rmd_loss.item()) * bsz
            totals["mlm"] += float(mlm_loss.item()) * bsz
            iterator.set_postfix(
                step=f"{global_step}/{total_steps}",
                loss=totals["loss"] / seen,
                rtd=totals["rtd"] / seen,
                rmd=totals["rmd"] / seen,
            )
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break
        epoch_summary = {k: v / max(seen, 1) for k, v in totals.items()}
        epoch_summary["steps"] = global_step
        history.append(epoch_summary)
        if stop_training:
            break
    return history


@torch.no_grad()
def score_queries(args, tokenizer, discriminator, texts: list[str], device) -> np.ndarray:
    discriminator.eval()
    scores = []
    for start in tqdm(range(0, len(texts), args.eval_batch_size), desc="DATE scoring"):
        batch_texts = texts[start : start + args.eval_batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        rtd_logits, _ = discriminator(input_ids, attention_mask)
        probs_original = torch.softmax(rtd_logits, dim=-1)[..., 0]
        special_masks = torch.as_tensor(
            [
                tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
                for row in encoded["input_ids"]
            ],
            dtype=torch.bool,
            device=device,
        )
        valid = (attention_mask == 1) & (~special_masks)
        denom = valid.sum(dim=1).clamp_min(1)
        pl_rtd = (probs_original * valid).sum(dim=1) / denom
        anomaly = 1.0 - pl_rtd
        scores.extend(anomaly.detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def aggregate_scores(scores: np.ndarray, args: argparse.Namespace) -> float:
    if len(scores) == 0:
        return float("nan")
    if args.batch_score_agg == "mean":
        return float(np.mean(scores))
    if args.batch_score_agg == "median":
        return float(np.median(scores))
    if args.batch_score_agg == "max":
        return float(np.max(scores))
    if args.batch_score_agg == "quantile":
        return float(np.quantile(scores, args.score_quantile))
    if args.batch_score_agg == "topk_mean":
        k = max(1, int(math.ceil(len(scores) * args.topk_fraction)))
        return float(np.mean(np.sort(scores)[-k:]))
    raise ValueError(f"Unknown aggregation: {args.batch_score_agg}")


def sample_scores(scores: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    replace = len(scores) < size
    idx = rng.choice(len(scores), size=size, replace=replace)
    return scores[idx]


def build_null_scores(benign_scores: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(args.null_samples, dtype=np.float64)
    for i in range(args.null_samples):
        values[i] = aggregate_scores(sample_scores(benign_scores, args.batch_size, rng), args)
    return values


def compute_thresholds(null_scores: np.ndarray, percentile: float, sidedness: str) -> dict[str, float]:
    if not 0.0 < percentile < 100.0:
        raise ValueError("--threshold-percentile must be between 0 and 100")
    if sidedness == "upper":
        return {"upper": float(np.percentile(null_scores, percentile))}
    if sidedness == "lower":
        return {"lower": float(np.percentile(null_scores, 100.0 - percentile))}
    if sidedness == "two-sided":
        alpha = 100.0 - percentile
        return {
            "lower": float(np.percentile(null_scores, alpha / 2.0)),
            "upper": float(np.percentile(null_scores, 100.0 - alpha / 2.0)),
        }
    raise ValueError(f"Unknown sidedness: {sidedness}")


def is_suspicious(score: float, thresholds: dict[str, float], sidedness: str) -> bool:
    if sidedness == "upper":
        return score > thresholds["upper"]
    if sidedness == "lower":
        return score < thresholds["lower"]
    if sidedness == "two-sided":
        return score < thresholds["lower"] or score > thresholds["upper"]
    raise ValueError(f"Unknown sidedness: {sidedness}")


def p_value(null_scores: np.ndarray, score: float, sidedness: str) -> float:
    n = len(null_scores)
    upper = (np.sum(null_scores >= score) + 1.0) / (n + 1.0)
    lower = (np.sum(null_scores <= score) + 1.0) / (n + 1.0)
    if sidedness == "upper":
        return float(upper)
    if sidedness == "lower":
        return float(lower)
    if sidedness == "two-sided":
        return float(min(1.0, 2.0 * min(lower, upper)))
    raise ValueError(f"Unknown sidedness: {sidedness}")


def evaluate_sequential(
    split: str,
    label: int,
    scores: np.ndarray,
    args: argparse.Namespace,
    null_scores: np.ndarray,
    thresholds: dict[str, float],
) -> list[BatchScore]:
    results = []
    for batch_id, batch in enumerate(iter_batches(scores.tolist(), args.batch_size, not args.keep_last_batch)):
        batch_scores = np.asarray(batch, dtype=np.float64)
        score = aggregate_scores(batch_scores, args)
        suspicious = is_suspicious(score, thresholds, args.test_sidedness)
        results.append(BatchScore(split, batch_id, label, len(batch), float(label), len(batch) if label else 0, score, p_value(null_scores, score, args.test_sidedness), suspicious))
    return results


def evaluate_random(
    split: str,
    label: int,
    scores: np.ndarray,
    num_batches: int,
    args: argparse.Namespace,
    null_scores: np.ndarray,
    thresholds: dict[str, float],
    rng: np.random.Generator,
) -> list[BatchScore]:
    results = []
    for batch_id in range(num_batches):
        batch_scores = sample_scores(scores, args.batch_size, rng)
        score = aggregate_scores(batch_scores, args)
        suspicious = is_suspicious(score, thresholds, args.test_sidedness)
        results.append(BatchScore(split, batch_id, label, args.batch_size, float(label), args.batch_size if label else 0, score, p_value(null_scores, score, args.test_sidedness), suspicious))
    return results


def evaluate_mixed(
    benign_scores: np.ndarray,
    attacker_scores: np.ndarray,
    ratios: list[float],
    args: argparse.Namespace,
    null_scores: np.ndarray,
    thresholds: dict[str, float],
    rng: np.random.Generator,
) -> list[BatchScore]:
    results = []
    for requested_ratio in ratios:
        attacker_count = int(round(args.batch_size * requested_ratio))
        attacker_count = min(max(attacker_count, 1), args.batch_size - 1)
        benign_count = args.batch_size - attacker_count
        actual_ratio = attacker_count / args.batch_size
        for batch_id in range(args.mixed_batches_per_ratio):
            batch_scores = np.concatenate([
                sample_scores(benign_scores, benign_count, rng),
                sample_scores(attacker_scores, attacker_count, rng),
            ])
            rng.shuffle(batch_scores)
            score = aggregate_scores(batch_scores, args)
            suspicious = is_suspicious(score, thresholds, args.test_sidedness)
            results.append(BatchScore(f"mixed_{requested_ratio:g}", batch_id, 1, args.batch_size, actual_ratio, attacker_count, score, p_value(null_scores, score, args.test_sidedness), suspicious))
    return results


def summarize(results: list[BatchScore]) -> dict[str, float | int]:
    positives = [r for r in results if r.label == 1]
    negatives = [r for r in results if r.label == 0]
    tp = sum(r.suspicious for r in positives)
    fp = sum(r.suspicious for r in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
    total = len(results)
    return {
        "batches": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / len(positives) if positives else float("nan"),
        "fpr": fp / len(negatives) if negatives else float("nan"),
        "accuracy": (tp + tn) / total if total else float("nan"),
    }


def summarize_by_split(results: list[BatchScore]) -> dict[str, dict[str, float | int]]:
    summaries = {}
    for split in sorted({r.split for r in results}):
        split_results = [r for r in results if r.split == split]
        suspicious = sum(r.suspicious for r in split_results)
        summaries[split] = {
            "batches": len(split_results),
            "suspicious": suspicious,
            "detection_rate": suspicious / len(split_results) if split_results else float("nan"),
            "mean_score": float(np.mean([r.score for r in split_results])) if split_results else float("nan"),
            "mean_attacker_ratio": float(np.mean([r.attacker_ratio for r in split_results])) if split_results else float("nan"),
        }
    return summaries


def write_outputs(output_dir: Path, results: list[BatchScore], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "batch_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "batch_id", "label", "size", "attacker_ratio", "attacker_count", "score", "p_value", "suspicious"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.generator == "learned" and not args.generator_model_name:
        if args.model_name == "google/electra-small-discriminator":
            args.generator_model_name = "google/electra-small-generator"
        else:
            args.generator_model_name = args.model_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    normal_path, attacker_path, resolved_dataset = resolve_paths(args)
    normal_queries = read_jsonl_queries(normal_path, args.text_field, args.max_normal)
    attacker_queries = read_jsonl_queries(attacker_path, args.text_field, args.max_attacker)
    rng.shuffle(normal_queries)
    rng.shuffle(attacker_queries)

    split_at = int(len(normal_queries) * args.normal_train_ratio)
    train_queries = normal_queries[:split_at]
    benign_test_queries = normal_queries[split_at:]
    if len(train_queries) < 2:
        raise ValueError("Need at least two normal training queries")
    if len(benign_test_queries) < 2:
        raise ValueError("Held-out benign test stream has fewer than two queries")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise ValueError(f"Tokenizer for {args.model_name} has no mask token")
    mask_patterns = generate_mask_patterns(args.mask_patterns, max(1, args.max_length - 2), args.mask_ratio, rng)
    collator = DateCollator(tokenizer, mask_patterns, args.max_length, rng)

    discriminator = DateDiscriminator(args.model_name, args.mask_patterns).to(device)
    generator = None
    if args.generator == "learned":
        generator = AutoModelForMaskedLM.from_pretrained(args.generator_model_name).to(device)

    history = train_date(args, tokenizer, discriminator, generator, collator, train_queries, device)

    if args.save_model_dir:
        args.save_model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(args.save_model_dir)
        torch.save(
            {
                "discriminator": discriminator.state_dict(),
                "mask_patterns": mask_patterns,
                "args": vars(args),
            },
            args.save_model_dir / "date_detector.pt",
        )

    train_scores = score_queries(args, tokenizer, discriminator, train_queries, device)
    benign_scores = score_queries(args, tokenizer, discriminator, benign_test_queries, device)
    attacker_scores = score_queries(args, tokenizer, discriminator, attacker_queries, device)
    null_scores = build_null_scores(train_scores, args, rng)
    thresholds = compute_thresholds(null_scores, args.threshold_percentile, args.test_sidedness)

    mixed_ratios = [float(item.strip()) for item in args.mixed_attacker_ratios.split(",") if item.strip()]
    for ratio in mixed_ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError("--mixed-attacker-ratios values must be between 0 and 1")

    results: list[BatchScore] = []
    if args.benign_eval_batches > 0:
        results.extend(evaluate_random("heldout_benign", 0, benign_scores, args.benign_eval_batches, args, null_scores, thresholds, rng))
    else:
        results.extend(evaluate_sequential("heldout_benign", 0, benign_scores, args, null_scores, thresholds))
    if args.attacker_eval_batches > 0:
        results.extend(evaluate_random("attacker", 1, attacker_scores, args.attacker_eval_batches, args, null_scores, thresholds, rng))
    else:
        results.extend(evaluate_sequential("attacker", 1, attacker_scores, args, null_scores, thresholds))
    if mixed_ratios:
        results.extend(evaluate_mixed(benign_scores, attacker_scores, mixed_ratios, args, null_scores, thresholds, rng))

    metadata = {
        "dataset": resolved_dataset,
        "data_root": str(args.data_root),
        "normal_source": args.normal_source,
        "global_normal_path": str(args.global_normal_path),
        "normal_path": str(normal_path),
        "attacker_path": str(attacker_path),
        "text_field": args.text_field,
        "model_name": args.model_name,
        "generator": args.generator,
        "generator_model_name": args.generator_model_name,
        "max_length": args.max_length,
        "mask_patterns": args.mask_patterns,
        "mask_ratio": args.mask_ratio,
        "epochs": args.epochs,
        "max_train_steps": args.max_train_steps,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "rtd_weight": args.rtd_weight,
        "rmd_weight": args.rmd_weight,
        "batch_size": args.batch_size,
        "threshold_percentile": args.threshold_percentile,
        "test_sidedness": args.test_sidedness,
        "thresholds": thresholds,
        "threshold": thresholds.get("upper", thresholds.get("lower")),
        "null_samples": args.null_samples,
        "batch_score_agg": args.batch_score_agg,
        "score_quantile": args.score_quantile,
        "topk_fraction": args.topk_fraction,
        "normal_queries": len(normal_queries),
        "train_normal_queries": len(train_queries),
        "heldout_benign_size": len(benign_test_queries),
        "attacker_queries": len(attacker_queries),
        "null_source": "train_normal_queries",
        "mixed_attacker_ratios": mixed_ratios,
        "mixed_batches_per_ratio": args.mixed_batches_per_ratio,
        "benign_eval_batches": args.benign_eval_batches,
        "attacker_eval_batches": args.attacker_eval_batches,
        "device": str(device),
        "seed": args.seed,
        "training_history": history,
        "metrics": summarize(results),
        "metrics_by_split": summarize_by_split(results),
    }
    write_outputs(args.output_dir, results, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir / 'batch_scores.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
