#!/usr/bin/env python3
"""Original DATE-style query-level anomaly detection.

This script reuses the DATE training pipeline but evaluates each query as one
instance, matching the paper's sample-level anomaly detection setting more
closely than the batch-level detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from date_data import DateCollator, QueryDataset, generate_mask_patterns, read_jsonl_queries
from date_detector import (
    compute_thresholds,
    is_suspicious as score_is_suspicious,
    p_value as empirical_p_value,
    resolve_paths,
    score_queries,
    train_date,
)
from date_model import DateDiscriminator


@dataclass
class QueryScore:
    split: str
    query_id: int
    label: int
    score: float
    p_value: float
    suspicious: bool
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run original DATE-style query-level normal/attacker detection."
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
    parser.add_argument("--generator-model-name", default=None)
    parser.add_argument("--generator", choices=["learned", "random"], default="learned")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mask-patterns", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-steps", type=int, default=5000)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--rtd-weight", type=float, default=50.0)
    parser.add_argument("--rmd-weight", type=float, default=100.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default=None)

    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument(
        "--test-sidedness",
        choices=["upper", "lower", "two-sided"],
        default="upper",
        help="Tail test for DATE scores. two-sided uses alpha split across both tails.",
    )
    parser.add_argument("--benign-eval-limit", type=int, default=None)
    parser.add_argument("--attacker-eval-limit", type=int, default=None)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("DATE/original_outputs"))
    parser.add_argument("--save-model-dir", type=Path, default=None)
    return parser.parse_args()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    pos_rank_sum = float(np.sum(ranks[positives]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    n_pos = int(np.sum(positives))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(labels) + 1)
    precision = tp / ranks
    return float(np.sum(precision[sorted_labels == 1]) / n_pos)


def summarize(labels: np.ndarray, scores: np.ndarray, suspicious: np.ndarray) -> dict[str, float | int]:
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(suspicious & positives))
    fp = int(np.sum(suspicious & negatives))
    fn = int(np.sum((~suspicious) & positives))
    tn = int(np.sum((~suspicious) & negatives))
    total = len(labels)
    return {
        "queries": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / int(np.sum(positives)) if np.any(positives) else float("nan"),
        "fpr": fp / int(np.sum(negatives)) if np.any(negatives) else float("nan"),
        "accuracy": (tp + tn) / total if total else float("nan"),
        "auroc": roc_auc(labels, scores),
        "auprc": average_precision(labels, scores),
    }


def write_outputs(output_dir: Path, rows: list[QueryScore], metadata: dict, include_text: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "query_id", "label", "score", "p_value", "suspicious"]
    if include_text:
        fieldnames.append("text")
    with (output_dir / "query_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "split": row.split,
                "query_id": row.query_id,
                "label": row.label,
                "score": row.score,
                "p_value": row.p_value,
                "suspicious": row.suspicious,
            }
            if include_text:
                record["text"] = row.text
            writer.writerow(record)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")
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
    if args.benign_eval_limit is not None:
        benign_test_queries = benign_test_queries[: args.benign_eval_limit]
    if args.attacker_eval_limit is not None:
        attacker_queries = attacker_queries[: args.attacker_eval_limit]
    if len(train_queries) < 2:
        raise ValueError("Need at least two normal training queries")
    if len(benign_test_queries) < 1:
        raise ValueError("No held-out benign queries to evaluate")
    if len(attacker_queries) < 1:
        raise ValueError("No attacker queries to evaluate")

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
            args.save_model_dir / "date_original_detector.pt",
        )

    train_scores = score_queries(args, tokenizer, discriminator, train_queries, device)
    benign_scores = score_queries(args, tokenizer, discriminator, benign_test_queries, device)
    attacker_scores = score_queries(args, tokenizer, discriminator, attacker_queries, device)
    thresholds = compute_thresholds(train_scores, args.threshold_percentile, args.test_sidedness)

    eval_scores = np.concatenate([benign_scores, attacker_scores])
    labels = np.concatenate([
        np.zeros(len(benign_scores), dtype=np.int64),
        np.ones(len(attacker_scores), dtype=np.int64),
    ])
    suspicious = np.asarray(
        [score_is_suspicious(float(score), thresholds, args.test_sidedness) for score in eval_scores],
        dtype=np.bool_,
    )
    p_values = np.asarray(
        [empirical_p_value(train_scores, float(score), args.test_sidedness) for score in eval_scores],
        dtype=np.float64,
    )

    rows: list[QueryScore] = []
    for i, (score, pval, suspicious_flag, text) in enumerate(
        zip(benign_scores, p_values[: len(benign_scores)], suspicious[: len(benign_scores)], benign_test_queries)
    ):
        rows.append(QueryScore("heldout_benign", i, 0, float(score), float(pval), bool(suspicious_flag), text))
    offset = len(benign_scores)
    for i, (score, pval, suspicious_flag, text) in enumerate(
        zip(attacker_scores, p_values[offset:], suspicious[offset:], attacker_queries)
    ):
        rows.append(QueryScore("attacker", i, 1, float(score), float(pval), bool(suspicious_flag), text))

    metadata = {
        "level": "query",
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
        "threshold_percentile": args.threshold_percentile,
        "test_sidedness": args.test_sidedness,
        "thresholds": thresholds,
        "threshold": thresholds.get("upper", thresholds.get("lower")),
        "null_source": "train_normal_queries",
        "normal_queries": len(normal_queries),
        "train_normal_queries": len(train_queries),
        "heldout_benign_queries": len(benign_test_queries),
        "attacker_queries": len(attacker_queries),
        "device": str(device),
        "seed": args.seed,
        "training_history": history,
        "metrics": summarize(labels, eval_scores, suspicious),
    }
    write_outputs(args.output_dir, rows, metadata, args.include_text)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir / 'query_scores.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
