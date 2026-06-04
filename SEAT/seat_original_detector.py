#!/usr/bin/env python3
"""Original SEAT-style account-level detection on LLM query embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

np = None


@dataclass
class AccountScore:
    split: str
    account_id: int
    label: int
    size: int
    attacker_ratio: float
    attacker_count: int
    suspicious: bool
    queries_until_alert: int
    similar_pair_count: int
    score: float


def read_jsonl_queries(path: Path, text_field: str, limit: int | None = None) -> list[str]:
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            queries.append(text.strip())
            if limit is not None and len(queries) >= limit:
                break
    if not queries:
        raise ValueError(f"No usable '{text_field}' strings found in {path}")
    return queries


def resolve_dataset_dir(data_root: Path, dataset: str) -> Path:
    exact = data_root / dataset
    if exact.exists():
        return exact
    if data_root.exists():
        matches = [p for p in data_root.iterdir() if p.is_dir() and p.name.lower() == dataset.lower()]
        if len(matches) == 1:
            return matches[0]
    available = []
    if data_root.exists():
        available = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    available_text = ", ".join(available) if available else "none"
    raise FileNotFoundError(
        f"Dataset '{dataset}' was not found under {data_root}. Available datasets: {available_text}"
    )


def default_dataset_paths(data_root: Path, dataset: str) -> tuple[Path, Path, str]:
    dataset_dir = resolve_dataset_dir(data_root, dataset)
    normal_path = dataset_dir / "normal" / "queries.jsonl"
    attacker_path = dataset_dir / "attacker" / "queries.jsonl"
    missing = [str(path) for path in (normal_path, attacker_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset must contain normal/queries.jsonl and attacker/queries.jsonl. "
            f"Missing: {', '.join(missing)}"
        )
    return normal_path, attacker_path, dataset_dir.name


def default_attacker_path(data_root: Path, dataset: str) -> tuple[Path, str]:
    dataset_dir = resolve_dataset_dir(data_root, dataset)
    attacker_path = dataset_dir / "attacker" / "queries.jsonl"
    if not attacker_path.exists():
        raise FileNotFoundError(f"Missing attacker queries: {attacker_path}")
    return attacker_path, dataset_dir.name


def cache_key(texts: list[str], model_name: str, prefix: str, normalize: bool) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\0")
    h.update(prefix.encode("utf-8"))
    h.update(b"\0")
    h.update(str(normalize).encode("ascii"))
    h.update(b"\0")
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str | None,
    prefix: str,
    normalize: bool,
    cache_dir: Path | None,
):
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key(texts, model_name, prefix, normalize)}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install it first, e.g. "
            "`pip install -r requirements.txt`."
        ) from exc

    model_kwargs = {}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(model_name, **model_kwargs)
    encoded = model.encode(
        [prefix + text for text in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    embeddings = np.asarray(encoded, dtype=np.float32)
    if cache_dir is not None:
        np.save(cache_path, embeddings)
    return embeddings


def l2_normalize(z):
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norms, 1e-12)


def move_to_compute_device(z, compute_device: str | None):
    if not compute_device or compute_device == "cpu":
        return z, "numpy"
    try:
        import torch
    except ImportError:
        return z, "numpy"
    if compute_device.startswith("cuda") and not torch.cuda.is_available():
        return z, "numpy"
    return torch.as_tensor(z, dtype=torch.float32, device=compute_device), "torch"


def is_torch_tensor(x: object) -> bool:
    return x.__class__.__module__.startswith("torch")


def take_rows(z, idx):
    if is_torch_tensor(z):
        import torch

        return z[torch.as_tensor(idx, dtype=torch.long, device=z.device)]
    return z[idx]


def concat_rows(parts):
    if parts and is_torch_tensor(parts[0]):
        import torch

        return torch.cat(parts, dim=0)
    return np.concatenate(parts, axis=0)


def sample_account(z, account_size: int, rng):
    replace = len(z) < account_size
    idx = rng.choice(len(z), size=account_size, replace=replace)
    return take_rows(z, idx)


def iter_accounts(z, account_size: int, drop_last: bool):
    for start in range(0, len(z), account_size):
        account = z[start : start + account_size]
        if len(account) < account_size and drop_last:
            continue
        if len(account) >= 2:
            yield account


def detect_account(z_account, similarity_threshold: float, similar_pair_threshold: int) -> tuple[bool, int, int]:
    sims = z_account @ z_account.T
    if is_torch_tensor(sims):
        import torch

        similar = torch.triu(sims >= similarity_threshold, diagonal=1)
        per_query_pairs = similar.sum(dim=0)
        cumulative = torch.cumsum(per_query_pairs, dim=0)
        total_pairs = int(cumulative[-1].item())
        alert_positions = torch.nonzero(cumulative > similar_pair_threshold, as_tuple=False)
        if len(alert_positions) > 0:
            idx = int(alert_positions[0].item())
            return True, idx + 1, int(cumulative[idx].item())
        return False, len(z_account), total_pairs

    similar = np.triu(sims >= similarity_threshold, k=1)
    per_query_pairs = similar.sum(axis=0)
    cumulative = np.cumsum(per_query_pairs)
    total_pairs = int(cumulative[-1])
    alert_positions = np.flatnonzero(cumulative > similar_pair_threshold)
    if len(alert_positions) > 0:
        idx = int(alert_positions[0])
        return True, idx + 1, int(cumulative[idx])
    return False, len(z_account), total_pairs


def sample_pair_similarities(z, sample_size: int, rng):
    if len(z) < 2:
        raise ValueError("Need at least two benign embeddings to sample pair similarities")
    first = rng.integers(0, len(z), size=sample_size)
    second = rng.integers(0, len(z) - 1, size=sample_size)
    second = np.where(second >= first, second + 1, second)
    first_rows = take_rows(z, first)
    second_rows = take_rows(z, second)
    if is_torch_tensor(z):
        return ((first_rows * second_rows).sum(dim=1)).detach().cpu().numpy()
    return np.sum(first_rows * second_rows, axis=1)


def calibrate_similarity_threshold_from_pairs(
    z_b,
    account_size: int,
    similar_pair_threshold: int,
    pair_sample_size: int,
    pair_tail_scale: float,
    rng,
) -> tuple[float, float]:
    total_pairs = account_size * (account_size - 1) / 2.0
    tail_probability = pair_tail_scale * (similar_pair_threshold + 1) / total_pairs
    tail_probability = min(max(tail_probability, 1.0 / pair_sample_size), 0.5)
    percentile = 100.0 * (1.0 - tail_probability)
    pair_sims = sample_pair_similarities(z_b, pair_sample_size, rng)
    threshold = float(np.percentile(pair_sims, percentile))
    threshold = min(max(threshold, -1.0), 1.0)
    return threshold, float(percentile)


def estimate_false_positive_rate(
    z_b,
    account_size: int,
    calibration_accounts: int,
    similarity_threshold: float,
    similar_pair_threshold: int,
    rng,
) -> float:
    alerts = 0
    for _ in range(calibration_accounts):
        account = sample_account(z_b, account_size, rng)
        suspicious, _, _ = detect_account(account, similarity_threshold, similar_pair_threshold)
        alerts += int(suspicious)
    return alerts / calibration_accounts


def calibrate_similarity_threshold(
    z_b,
    account_size: int,
    calibration_accounts: int,
    target_fpr: float,
    similar_pair_threshold: int,
    binary_search_steps: int,
    rng,
) -> tuple[float, float]:
    lo, hi = -1.0, 1.0
    best_threshold = hi
    best_fpr = 1.0
    for _ in range(binary_search_steps):
        mid = (lo + hi) / 2.0
        fpr = estimate_false_positive_rate(
            z_b,
            account_size,
            calibration_accounts,
            mid,
            similar_pair_threshold,
            rng,
        )
        if fpr <= target_fpr:
            best_threshold = mid
            best_fpr = fpr
            hi = mid
        else:
            lo = mid
    return float(best_threshold), float(best_fpr)


def evaluate_accounts(
    split: str,
    label: int,
    z_stream,
    account_size: int,
    eval_accounts: int,
    random_accounts: bool,
    drop_last: bool,
    similarity_threshold: float,
    similar_pair_threshold: int,
    rng,
) -> list[AccountScore]:
    results: list[AccountScore] = []
    if random_accounts:
        accounts = [sample_account(z_stream, account_size, rng) for _ in range(eval_accounts)]
    else:
        accounts = list(iter_accounts(z_stream, account_size, drop_last))
        if eval_accounts > 0:
            accounts = accounts[:eval_accounts]

    for account_id, account in enumerate(accounts):
        suspicious, queries_until_alert, similar_pair_count = detect_account(
            account, similarity_threshold, similar_pair_threshold
        )
        results.append(
            AccountScore(
                split=split,
                account_id=account_id,
                label=label,
                size=len(account),
                attacker_ratio=float(label),
                attacker_count=len(account) if label == 1 else 0,
                suspicious=suspicious,
                queries_until_alert=queries_until_alert,
                similar_pair_count=similar_pair_count,
                score=float(similar_pair_count),
            )
        )
    return results


def evaluate_mixed_accounts(
    z_benign_stream,
    z_attacker_stream,
    account_size: int,
    attacker_ratios: list[float],
    accounts_per_ratio: int,
    similarity_threshold: float,
    similar_pair_threshold: int,
    rng,
) -> list[AccountScore]:
    results: list[AccountScore] = []
    for requested_ratio in attacker_ratios:
        attacker_count = int(round(account_size * requested_ratio))
        attacker_count = min(max(attacker_count, 1), account_size - 1)
        benign_count = account_size - attacker_count
        actual_ratio = attacker_count / account_size
        for account_id in range(accounts_per_ratio):
            benign_part = sample_account(z_benign_stream, benign_count, rng)
            attacker_part = sample_account(z_attacker_stream, attacker_count, rng)
            account = concat_rows([benign_part, attacker_part])
            account = take_rows(account, rng.permutation(len(account)))
            suspicious, queries_until_alert, similar_pair_count = detect_account(
                account, similarity_threshold, similar_pair_threshold
            )
            results.append(
                AccountScore(
                    split=f"mixed_{requested_ratio:g}",
                    account_id=account_id,
                    label=1,
                    size=len(account),
                    attacker_ratio=actual_ratio,
                    attacker_count=attacker_count,
                    suspicious=suspicious,
                    queries_until_alert=queries_until_alert,
                    similar_pair_count=similar_pair_count,
                    score=float(similar_pair_count),
                )
            )
    return results


def summarize(results: list[AccountScore]) -> dict[str, float | int]:
    total = len(results)
    positives = [r for r in results if r.label == 1]
    negatives = [r for r in results if r.label == 0]
    tp = sum(r.suspicious for r in positives)
    fp = sum(r.suspicious for r in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
    return {
        "accounts": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / len(positives) if positives else float("nan"),
        "fpr": fp / len(negatives) if negatives else float("nan"),
        "accuracy": (tp + tn) / total if total else float("nan"),
    }


def summarize_by_split(results: list[AccountScore]) -> dict[str, dict[str, float | int]]:
    summaries = {}
    for split in sorted({r.split for r in results}):
        split_results = [r for r in results if r.split == split]
        suspicious = sum(r.suspicious for r in split_results)
        alerted = [r.queries_until_alert for r in split_results if r.suspicious]
        summaries[split] = {
            "accounts": len(split_results),
            "suspicious": suspicious,
            "detection_rate": suspicious / len(split_results) if split_results else float("nan"),
            "mean_score": float(np.mean([r.score for r in split_results])) if split_results else float("nan"),
            "mean_queries_until_alert": float(np.mean(alerted)) if alerted else float("nan"),
            "mean_attacker_ratio": (
                float(np.mean([r.attacker_ratio for r in split_results]))
                if split_results
                else float("nan")
            ),
        }
    return summaries


def write_outputs(output_dir: Path, results: list[AccountScore], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "account_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "account_id",
                "label",
                "size",
                "attacker_ratio",
                "attacker_count",
                "suspicious",
                "queries_until_alert",
                "similar_pair_count",
                "score",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "split": r.split,
                    "account_id": r.account_id,
                    "label": r.label,
                    "size": r.size,
                    "attacker_ratio": r.attacker_ratio,
                    "attacker_count": r.attacker_count,
                    "suspicious": r.suspicious,
                    "queries_until_alert": r.queries_until_alert,
                    "similar_pair_count": r.similar_pair_count,
                    "score": r.score,
                }
            )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original SEAT-style account-level detection with similar-pair counting."
    )
    parser.add_argument("--dataset", default="model_leeching")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--normal-source",
        choices=["dataset", "global"],
        default="dataset",
        help="Use the dataset's own normal queries or --global-normal-path.",
    )
    parser.add_argument("--global-normal-path", type=Path, default=Path("data/global_normal/queries.jsonl"))
    parser.add_argument("--normal-path", type=Path, default=None)
    parser.add_argument("--attacker-path", type=Path, default=None)
    parser.add_argument("--text-field", default="query")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--device", default=None, help="SentenceTransformer device, e.g. cuda or cpu")
    parser.add_argument(
        "--compute-device",
        default=None,
        help="Device for pairwise similarity scoring. Defaults to --device when set.",
    )
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--account-size", type=int, default=1500)
    parser.add_argument(
        "--similar-pair-threshold",
        type=int,
        default=50,
        help="Original SEAT-style N_thresh. Alert after more than this many similar pairs.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Cosine threshold delta for similar pairs. If omitted, calibrate on benign accounts.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["pair-percentile", "account-fpr"],
        default="pair-percentile",
        help=(
            "pair-percentile is the fast default; account-fpr binary-searches account alerts "
            "and is closer to the paper but much slower."
        ),
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.0001,
        help="Target account-level FPR for calibrating delta when --similarity-threshold is omitted.",
    )
    parser.add_argument("--calibration-accounts", type=int, default=50)
    parser.add_argument("--binary-search-steps", type=int, default=16)
    parser.add_argument("--pair-sample-size", type=int, default=200000)
    parser.add_argument(
        "--pair-tail-scale",
        type=float,
        default=0.25,
        help="Conservative multiplier for fast pair-percentile calibration.",
    )
    parser.add_argument("--normal-train-ratio", type=float, default=0.8)
    parser.add_argument("--max-normal", type=int, default=None)
    parser.add_argument("--max-attacker", type=int, default=None)
    parser.add_argument(
        "--mixed-attacker-ratios",
        default="",
        help="Comma-separated attacker ratios for mixed accounts. Use empty string to disable.",
    )
    parser.add_argument("--mixed-accounts-per-ratio", type=int, default=50)
    parser.add_argument("--benign-eval-accounts", type=int, default=50)
    parser.add_argument("--attacker-eval-accounts", type=int, default=50)
    parser.add_argument(
        "--sequential-eval",
        action="store_true",
        help="Evaluate sequential non-overlapping accounts instead of random sampled accounts.",
    )
    parser.add_argument("--keep-last-account", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("SEAT/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("SEAT/original_outputs"))
    return parser.parse_args()


def main() -> None:
    global np
    args = parse_args()
    try:
        import numpy as _np
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: numpy. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    np = _np

    if args.account_size < 2:
        raise ValueError("--account-size must be at least 2")
    if args.similar_pair_threshold < 0:
        raise ValueError("--similar-pair-threshold must be non-negative")
    if not 0.0 <= args.target_fpr <= 1.0:
        raise ValueError("--target-fpr must be between 0 and 1")
    if args.calibration_accounts < 1:
        raise ValueError("--calibration-accounts must be at least 1")
    if args.binary_search_steps < 1:
        raise ValueError("--binary-search-steps must be at least 1")
    if args.pair_sample_size < 1:
        raise ValueError("--pair-sample-size must be at least 1")
    if args.pair_tail_scale <= 0:
        raise ValueError("--pair-tail-scale must be positive")
    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    resolved_dataset = args.dataset
    if args.normal_path is None and args.attacker_path is None and args.normal_source == "dataset":
        normal_path, attacker_path, resolved_dataset = default_dataset_paths(
            args.data_root, args.dataset
        )
    else:
        if args.attacker_path is not None:
            attacker_path = args.attacker_path
        else:
            attacker_path, resolved_dataset = default_attacker_path(args.data_root, args.dataset)
        if args.normal_path is not None:
            normal_path = args.normal_path
        elif args.normal_source == "global":
            normal_path = args.global_normal_path
        else:
            normal_path = args.data_root / resolved_dataset / "normal" / "queries.jsonl"
        if not normal_path.exists():
            raise FileNotFoundError(
                f"Missing normal queries: {normal_path}. "
                "Use --normal-source global for attacker-only datasets."
            )

    mixed_ratios = [
        float(item.strip())
        for item in args.mixed_attacker_ratios.split(",")
        if item.strip()
    ]
    for ratio in mixed_ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError("--mixed-attacker-ratios values must be between 0 and 1")

    normal_queries = read_jsonl_queries(normal_path, args.text_field, args.max_normal)
    attacker_queries = read_jsonl_queries(attacker_path, args.text_field, args.max_attacker)
    rng.shuffle(normal_queries)
    rng.shuffle(attacker_queries)

    split_at = int(len(normal_queries) * args.normal_train_ratio)
    benign_pool_queries = normal_queries[:split_at]
    benign_test_queries = normal_queries[split_at:]
    if len(benign_pool_queries) < 2:
        raise ValueError("Benign reference pool has fewer than 2 queries")
    if len(benign_test_queries) < 2:
        raise ValueError("Held-out benign test stream has fewer than 2 queries")
    if len(attacker_queries) < 2:
        raise ValueError("Attacker stream has fewer than 2 queries")

    all_queries = benign_pool_queries + benign_test_queries + attacker_queries
    embeddings = embed_texts(
        all_queries,
        model_name=args.embedding_model,
        batch_size=args.encode_batch_size,
        device=args.device,
        prefix=args.query_prefix,
        normalize=not args.no_normalize_embeddings,
        cache_dir=args.cache_dir,
    )
    embeddings = l2_normalize(embeddings)
    embeddings, compute_backend = move_to_compute_device(embeddings, args.compute_device or args.device)

    n_b = len(benign_pool_queries)
    n_h = len(benign_test_queries)
    z_b = embeddings[:n_b]
    z_benign_test = embeddings[n_b : n_b + n_h]
    z_attacker = embeddings[n_b + n_h :]

    if args.similarity_threshold is None:
        if args.calibration_mode == "account-fpr":
            similarity_threshold, calibration_fpr = calibrate_similarity_threshold(
                z_b=z_b,
                account_size=args.account_size,
                calibration_accounts=args.calibration_accounts,
                target_fpr=args.target_fpr,
                similar_pair_threshold=args.similar_pair_threshold,
                binary_search_steps=args.binary_search_steps,
                rng=rng,
            )
            pair_percentile = None
            similarity_threshold_source = "calibrated_account_fpr"
        else:
            similarity_threshold, pair_percentile = calibrate_similarity_threshold_from_pairs(
                z_b=z_b,
                account_size=args.account_size,
                similar_pair_threshold=args.similar_pair_threshold,
                pair_sample_size=args.pair_sample_size,
                pair_tail_scale=args.pair_tail_scale,
                rng=rng,
            )
            calibration_fpr = estimate_false_positive_rate(
                z_b=z_b,
                account_size=args.account_size,
                calibration_accounts=args.calibration_accounts,
                similarity_threshold=similarity_threshold,
                similar_pair_threshold=args.similar_pair_threshold,
                rng=rng,
            )
            similarity_threshold_source = "calibrated_pair_percentile"
    else:
        similarity_threshold = float(args.similarity_threshold)
        pair_percentile = None
        calibration_fpr = estimate_false_positive_rate(
            z_b=z_b,
            account_size=args.account_size,
            calibration_accounts=args.calibration_accounts,
            similarity_threshold=similarity_threshold,
            similar_pair_threshold=args.similar_pair_threshold,
            rng=rng,
        )
        similarity_threshold_source = "manual"

    random_accounts = not args.sequential_eval
    results = []
    results.extend(
        evaluate_accounts(
            split="heldout_benign",
            label=0,
            z_stream=z_benign_test,
            account_size=args.account_size,
            eval_accounts=args.benign_eval_accounts,
            random_accounts=random_accounts,
            drop_last=not args.keep_last_account,
            similarity_threshold=similarity_threshold,
            similar_pair_threshold=args.similar_pair_threshold,
            rng=rng,
        )
    )
    results.extend(
        evaluate_accounts(
            split="attacker",
            label=1,
            z_stream=z_attacker,
            account_size=args.account_size,
            eval_accounts=args.attacker_eval_accounts,
            random_accounts=random_accounts,
            drop_last=not args.keep_last_account,
            similarity_threshold=similarity_threshold,
            similar_pair_threshold=args.similar_pair_threshold,
            rng=rng,
        )
    )
    if mixed_ratios:
        results.extend(
            evaluate_mixed_accounts(
                z_benign_stream=z_benign_test,
                z_attacker_stream=z_attacker,
                account_size=args.account_size,
                attacker_ratios=mixed_ratios,
                accounts_per_ratio=args.mixed_accounts_per_ratio,
                similarity_threshold=similarity_threshold,
                similar_pair_threshold=args.similar_pair_threshold,
                rng=rng,
            )
        )

    metrics = summarize(results)
    metrics_by_split = summarize_by_split(results)
    metadata = {
        "dataset": resolved_dataset,
        "data_root": str(args.data_root),
        "normal_source": args.normal_source,
        "global_normal_path": str(args.global_normal_path),
        "normal_path": str(normal_path),
        "attacker_path": str(attacker_path),
        "text_field": args.text_field,
        "embedding_model": args.embedding_model,
        "embedding_device": args.device,
        "compute_device": args.compute_device or args.device or "cpu",
        "compute_backend": compute_backend,
        "query_prefix": args.query_prefix,
        "normalize_embeddings": not args.no_normalize_embeddings,
        "account_size": args.account_size,
        "similar_pair_threshold": args.similar_pair_threshold,
        "similarity_threshold": similarity_threshold,
        "similarity_threshold_source": similarity_threshold_source,
        "calibration_mode": args.calibration_mode,
        "pair_percentile": pair_percentile,
        "pair_sample_size": args.pair_sample_size,
        "pair_tail_scale": args.pair_tail_scale,
        "target_fpr": args.target_fpr,
        "calibration_accounts": args.calibration_accounts,
        "binary_search_steps": args.binary_search_steps,
        "calibration_fpr": calibration_fpr,
        "normal_queries": len(normal_queries),
        "benign_pool_size": len(benign_pool_queries),
        "heldout_benign_size": len(benign_test_queries),
        "attacker_queries": len(attacker_queries),
        "mixed_attacker_ratios": mixed_ratios,
        "mixed_accounts_per_ratio": args.mixed_accounts_per_ratio,
        "benign_eval_accounts": args.benign_eval_accounts,
        "attacker_eval_accounts": args.attacker_eval_accounts,
        "sequential_eval": args.sequential_eval,
        "seed": args.seed,
        "metrics": metrics,
        "metrics_by_split": metrics_by_split,
    }
    write_outputs(args.output_dir, results, metadata)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir / 'account_scores.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
