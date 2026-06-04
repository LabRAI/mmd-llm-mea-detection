#!/usr/bin/env python3
"""Batch-level PRADA-inspired detection on query embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

np = None

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = object


@dataclass
class BatchScore:
    split: str
    batch_id: int
    label: int
    size: int
    attacker_ratio: float
    attacker_count: int
    score: float
    w_statistic: float
    p_value: float
    suspicious: bool
    distances_used: int


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
) -> NDArray:
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
            "`pip install sentence-transformers`."
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


def pairwise_distances(x: NDArray, metric: str, device: str) -> NDArray:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch.") from exc

    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    if metric == "cosine":
        z = torch.nn.functional.normalize(xt, p=2, dim=1, eps=1e-12)
        distances = 1.0 - z @ z.T
        return torch.clamp(distances, min=0.0).detach().cpu().numpy()
    if metric == "l2":
        return torch.cdist(xt, xt, p=2).detach().cpu().numpy()
    raise ValueError(f"Unsupported distance metric: {metric}")


def nearest_neighbor_distances(x: NDArray, metric: str, device: str) -> NDArray:
    if len(x) < 2:
        raise ValueError("Need at least 2 samples to compute nearest-neighbor distances")
    distances = pairwise_distances(x, metric, device)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def trim_outliers_3sigma(values: NDArray) -> NDArray:
    if len(values) < 3:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return values
    mask = (values >= mean - 3.0 * std) & (values <= mean + 3.0 * std)
    trimmed = values[mask]
    return trimmed if len(trimmed) >= 3 else values


def subsample_for_shapiro(values: NDArray, max_samples: int, rng: object) -> NDArray:
    if max_samples <= 0 or len(values) <= max_samples:
        return values
    idx = rng.choice(len(values), size=max_samples, replace=False)
    return values[idx]


def shapiro_w(values: NDArray) -> float:
    try:
        from scipy.stats import shapiro
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: scipy. Install it in the research environment, e.g. "
            "`conda install -n research scipy` or `pip install scipy`."
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, _ = shapiro(values)
    return float(stat)


def prada_batch_score(
    batch: NDArray,
    metric: str,
    compute_device: str,
    max_shapiro_samples: int,
    rng: object,
) -> tuple[float, float, int]:
    distances = nearest_neighbor_distances(batch, metric, compute_device)
    distances = trim_outliers_3sigma(distances)
    distances = subsample_for_shapiro(distances, max_shapiro_samples, rng)
    if len(distances) < 3:
        raise ValueError("Need at least 3 nearest-neighbor distances for Shapiro-Wilk")
    w_statistic = shapiro_w(distances)
    score = 1.0 - w_statistic
    return score, w_statistic, len(distances)


def sample_batch(z: NDArray, size: int, rng: object) -> NDArray:
    replace = len(z) < size
    idx = rng.choice(len(z), size=size, replace=replace)
    return z[idx]


def iter_batches(z: NDArray, batch_size: int, drop_last: bool) -> Iterable[NDArray]:
    for start in range(0, len(z), batch_size):
        batch = z[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        if len(batch) >= 3:
            yield batch


def build_null_distribution(
    z_b: NDArray,
    batch_size: int,
    null_samples: int,
    metric: str,
    compute_device: str,
    max_shapiro_samples: int,
    rng: object,
) -> NDArray:
    scores = np.empty(null_samples, dtype=np.float64)
    for i in range(null_samples):
        batch = sample_batch(z_b, batch_size, rng)
        score, _, _ = prada_batch_score(batch, metric, compute_device, max_shapiro_samples, rng)
        scores[i] = score
    return scores


def score_batch(
    batch: NDArray,
    metric: str,
    compute_device: str,
    max_shapiro_samples: int,
    null_scores: NDArray,
    threshold_low: float,
    threshold_high: float,
    tail: str,
    rng: object,
) -> tuple[float, float, float, bool, int]:
    score, w_statistic, distances_used = prada_batch_score(
        batch, metric, compute_device, max_shapiro_samples, rng
    )
    lower_tail = float((np.sum(null_scores <= score) + 1.0) / (len(null_scores) + 1.0))
    upper_tail = float((np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0))
    if tail == "upper":
        p_value = upper_tail
        suspicious = score > threshold_high
    elif tail == "lower":
        p_value = lower_tail
        suspicious = score < threshold_low
    elif tail == "two-sided":
        p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
        suspicious = score < threshold_low or score > threshold_high
    else:
        raise ValueError(f"Unsupported tail: {tail}")
    return score, w_statistic, p_value, bool(suspicious), distances_used


def evaluate_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    batch_size: int,
    drop_last: bool,
    metric: str,
    compute_device: str,
    max_shapiro_samples: int,
    null_scores: NDArray,
    threshold_low: float,
    threshold_high: float,
    tail: str,
    rng: object,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for batch_id, batch in enumerate(iter_batches(z_stream, batch_size, drop_last)):
        score, w_statistic, p_value, suspicious, distances_used = score_batch(
            batch,
            metric,
            compute_device,
            max_shapiro_samples,
            null_scores,
            threshold_low,
            threshold_high,
            tail,
            rng,
        )
        results.append(
            BatchScore(
                split=split,
                batch_id=batch_id,
                label=label,
                size=len(batch),
                attacker_ratio=float(label),
                attacker_count=len(batch) if label == 1 else 0,
                score=score,
                w_statistic=w_statistic,
                p_value=p_value,
                suspicious=suspicious,
                distances_used=distances_used,
            )
        )
    return results


def evaluate_random_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    batch_size: int,
    num_batches: int,
    metric: str,
    compute_device: str,
    max_shapiro_samples: int,
    null_scores: NDArray,
    threshold_low: float,
    threshold_high: float,
    tail: str,
    rng: object,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for batch_id in range(num_batches):
        batch = sample_batch(z_stream, batch_size, rng)
        score, w_statistic, p_value, suspicious, distances_used = score_batch(
            batch,
            metric,
            compute_device,
            max_shapiro_samples,
            null_scores,
            threshold_low,
            threshold_high,
            tail,
            rng,
        )
        results.append(
            BatchScore(
                split=split,
                batch_id=batch_id,
                label=label,
                size=len(batch),
                attacker_ratio=float(label),
                attacker_count=len(batch) if label == 1 else 0,
                score=score,
                w_statistic=w_statistic,
                p_value=p_value,
                suspicious=suspicious,
                distances_used=distances_used,
            )
        )
    return results


def evaluate_mixed_stream(
    z_benign_stream: NDArray,
    z_attacker_stream: NDArray,
    batch_size: int,
    attacker_ratios: list[float],
    batches_per_ratio: int,
    metric: str,
    compute_device: str,
    max_shapiro_samples: int,
    null_scores: NDArray,
    threshold_low: float,
    threshold_high: float,
    tail: str,
    rng: object,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for requested_ratio in attacker_ratios:
        attacker_count = int(round(batch_size * requested_ratio))
        attacker_count = min(max(attacker_count, 1), batch_size - 1)
        benign_count = batch_size - attacker_count
        actual_ratio = attacker_count / batch_size
        for batch_id in range(batches_per_ratio):
            benign_part = sample_batch(z_benign_stream, benign_count, rng)
            attacker_part = sample_batch(z_attacker_stream, attacker_count, rng)
            batch = np.concatenate([benign_part, attacker_part], axis=0)
            batch = batch[rng.permutation(len(batch))]
            score, w_statistic, p_value, suspicious, distances_used = score_batch(
                batch,
                metric,
                compute_device,
                max_shapiro_samples,
                null_scores,
                threshold_low,
                threshold_high,
                tail,
                rng,
            )
            results.append(
                BatchScore(
                    split=f"mixed_{requested_ratio:g}",
                    batch_id=batch_id,
                    label=1,
                    size=len(batch),
                    attacker_ratio=actual_ratio,
                    attacker_count=attacker_count,
                    score=score,
                    w_statistic=w_statistic,
                    p_value=p_value,
                    suspicious=suspicious,
                    distances_used=distances_used,
                )
            )
    return results


def summarize(results: list[BatchScore]) -> dict[str, float | int]:
    total = len(results)
    positives = [r for r in results if r.label == 1]
    negatives = [r for r in results if r.label == 0]
    tp = sum(r.suspicious for r in positives)
    fp = sum(r.suspicious for r in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
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
            "mean_w_statistic": (
                float(np.mean([r.w_statistic for r in split_results])) if split_results else float("nan")
            ),
            "mean_attacker_ratio": (
                float(np.mean([r.attacker_ratio for r in split_results]))
                if split_results
                else float("nan")
            ),
        }
    return summaries


def write_outputs(output_dir: Path, results: list[BatchScore], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "batch_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "batch_id",
                "label",
                "size",
                "attacker_ratio",
                "attacker_count",
                "score",
                "w_statistic",
                "p_value",
                "suspicious",
                "distances_used",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "split": r.split,
                    "batch_id": r.batch_id,
                    "label": r.label,
                    "size": r.size,
                    "attacker_ratio": r.attacker_ratio,
                    "attacker_count": r.attacker_count,
                    "score": r.score,
                    "w_statistic": r.w_statistic,
                    "p_value": r.p_value,
                    "suspicious": r.suspicious,
                    "distances_used": r.distances_used,
                }
            )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect attacker query batches with a PRADA-inspired nearest-neighbor normality test."
    )
    parser.add_argument(
        "--dataset",
        default="model_leeching",
        help="Dataset folder under --data-root, e.g. model_leeching or Meaeq.",
    )
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
    parser.add_argument("--device", default="cuda", help="SentenceTransformer device, e.g. cuda or cpu")
    parser.add_argument(
        "--compute-device",
        default=None,
        help="Torch device for nearest-neighbor distance computation. Defaults to --device.",
    )
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--null-samples", type=int, default=1000)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--normal-train-ratio", type=float, default=0.8)
    parser.add_argument("--distance-metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument(
        "--tail",
        choices=["upper", "lower", "two-sided"],
        default="upper",
        help=(
            "Which side of the benign score distribution is suspicious. "
            "upper matches the original PRADA intuition: low W / high 1-W. "
            "two-sided also flags batches that look unusually Gaussian."
        ),
    )
    parser.add_argument(
        "--max-shapiro-samples",
        type=int,
        default=5000,
        help="Subsample nearest-neighbor distances before Shapiro-Wilk; 0 disables subsampling.",
    )
    parser.add_argument("--max-normal", type=int, default=None)
    parser.add_argument("--max-attacker", type=int, default=None)
    parser.add_argument(
        "--mixed-attacker-ratios",
        default="0.05,0.1,0.25,0.5",
        help="Comma-separated attacker ratios for mixed incoming batches. Use empty string to disable.",
    )
    parser.add_argument("--mixed-batches-per-ratio", type=int, default=50)
    parser.add_argument(
        "--benign-eval-batches",
        type=int,
        default=0,
        help="Random held-out benign batches to evaluate. Use 0 for sequential slicing.",
    )
    parser.add_argument(
        "--attacker-eval-batches",
        type=int,
        default=0,
        help="Random pure attacker batches to evaluate. Use 0 for sequential slicing.",
    )
    parser.add_argument("--keep-last-batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("PRADA/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("PRADA/outputs"))
    return parser.parse_args()


def main() -> None:
    global np
    args = parse_args()
    try:
        import numpy as _np
    except ImportError as exc:
        raise SystemExit("Missing dependency: numpy.") from exc
    np = _np

    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")
    if args.batch_size < 3:
        raise ValueError("--batch-size must be at least 3")
    if not 0.0 < args.threshold_percentile < 100.0:
        raise ValueError("--threshold-percentile must be between 0 and 100")

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    compute_device = args.compute_device or args.device

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
    if len(benign_pool_queries) < args.batch_size:
        raise ValueError("Benign reference pool is smaller than --batch-size")
    if len(benign_test_queries) < 3:
        raise ValueError("Held-out benign test stream has fewer than 3 queries")

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

    n_b = len(benign_pool_queries)
    n_h = len(benign_test_queries)
    z_b = embeddings[:n_b]
    z_benign_test = embeddings[n_b : n_b + n_h]
    z_attacker = embeddings[n_b + n_h :]

    null_scores = build_null_distribution(
        z_b=z_b,
        batch_size=args.batch_size,
        null_samples=args.null_samples,
        metric=args.distance_metric,
        compute_device=compute_device,
        max_shapiro_samples=args.max_shapiro_samples,
        rng=rng,
    )
    if args.tail == "two-sided":
        alpha = 100.0 - args.threshold_percentile
        threshold_low = float(np.percentile(null_scores, alpha / 2.0))
        threshold_high = float(np.percentile(null_scores, 100.0 - alpha / 2.0))
    elif args.tail == "lower":
        threshold_low = float(np.percentile(null_scores, 100.0 - args.threshold_percentile))
        threshold_high = float("inf")
    else:
        threshold_low = float("-inf")
        threshold_high = float(np.percentile(null_scores, args.threshold_percentile))
    threshold_low_w = 1.0 - threshold_low if threshold_low != float("-inf") else float("inf")
    threshold_high_w = 1.0 - threshold_high if threshold_high != float("inf") else float("-inf")

    results = []
    if args.benign_eval_batches > 0:
        results.extend(
            evaluate_random_stream(
                split="heldout_benign",
                label=0,
                z_stream=z_benign_test,
                batch_size=args.batch_size,
                num_batches=args.benign_eval_batches,
                metric=args.distance_metric,
                compute_device=compute_device,
                max_shapiro_samples=args.max_shapiro_samples,
                null_scores=null_scores,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                tail=args.tail,
                rng=rng,
            )
        )
    else:
        results.extend(
            evaluate_stream(
                split="heldout_benign",
                label=0,
                z_stream=z_benign_test,
                batch_size=args.batch_size,
                drop_last=not args.keep_last_batch,
                metric=args.distance_metric,
                compute_device=compute_device,
                max_shapiro_samples=args.max_shapiro_samples,
                null_scores=null_scores,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                tail=args.tail,
                rng=rng,
            )
        )

    if args.attacker_eval_batches > 0:
        results.extend(
            evaluate_random_stream(
                split="attacker",
                label=1,
                z_stream=z_attacker,
                batch_size=args.batch_size,
                num_batches=args.attacker_eval_batches,
                metric=args.distance_metric,
                compute_device=compute_device,
                max_shapiro_samples=args.max_shapiro_samples,
                null_scores=null_scores,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                tail=args.tail,
                rng=rng,
            )
        )
    else:
        results.extend(
            evaluate_stream(
                split="attacker",
                label=1,
                z_stream=z_attacker,
                batch_size=args.batch_size,
                drop_last=not args.keep_last_batch,
                metric=args.distance_metric,
                compute_device=compute_device,
                max_shapiro_samples=args.max_shapiro_samples,
                null_scores=null_scores,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                tail=args.tail,
                rng=rng,
            )
        )

    if mixed_ratios:
        results.extend(
            evaluate_mixed_stream(
                z_benign_stream=z_benign_test,
                z_attacker_stream=z_attacker,
                batch_size=args.batch_size,
                attacker_ratios=mixed_ratios,
                batches_per_ratio=args.mixed_batches_per_ratio,
                metric=args.distance_metric,
                compute_device=compute_device,
                max_shapiro_samples=args.max_shapiro_samples,
                null_scores=null_scores,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                tail=args.tail,
                rng=rng,
            )
        )

    metrics = summarize(results)
    metrics_by_split = summarize_by_split(results)
    metadata = {
        "method": "batch_prada_inspired_nn_distance_shapiro",
        "dataset": resolved_dataset,
        "data_root": str(args.data_root),
        "normal_source": args.normal_source,
        "global_normal_path": str(args.global_normal_path),
        "normal_path": str(normal_path),
        "attacker_path": str(attacker_path),
        "text_field": args.text_field,
        "embedding_model": args.embedding_model,
        "device": args.device,
        "compute_device": compute_device,
        "query_prefix": args.query_prefix,
        "normalize_embeddings": not args.no_normalize_embeddings,
        "batch_size": args.batch_size,
        "null_samples": args.null_samples,
        "threshold_percentile": args.threshold_percentile,
        "tail": args.tail,
        "threshold_low_score": threshold_low,
        "threshold_high_score": threshold_high,
        "threshold_low_w_statistic": threshold_low_w,
        "threshold_high_w_statistic": threshold_high_w,
        "distance_metric": args.distance_metric,
        "max_shapiro_samples": args.max_shapiro_samples,
        "normal_queries": len(normal_queries),
        "benign_pool_size": len(benign_pool_queries),
        "heldout_benign_size": len(benign_test_queries),
        "attacker_queries": len(attacker_queries),
        "mixed_attacker_ratios": mixed_ratios,
        "mixed_batches_per_ratio": args.mixed_batches_per_ratio,
        "benign_eval_batches": args.benign_eval_batches,
        "attacker_eval_batches": args.attacker_eval_batches,
        "seed": args.seed,
        "metrics": metrics,
        "metrics_by_split": metrics_by_split,
    }
    write_outputs(args.output_dir, results, metadata)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir / 'batch_scores.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
