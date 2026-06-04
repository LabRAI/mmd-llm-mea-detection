#!/usr/bin/env python3
"""Batch-level attacker detection with marginal Mahalanobis distance.

This implements the class-agnostic "marginal Mahalanobis distance" variant
from Podolskiy et al., Revisiting Mahalanobis Distance for Transformer-Based
Out-of-Domain Detection. Unlike class-conditional Mahalanobis, this detector
fits one benign centroid and covariance model, so it matches datasets without
internal benign class labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

np = None

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = object

Aggregation = Literal["mean", "median", "p90", "max"]


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


@dataclass
class MarginalMahalanobisModel:
    mean: NDArray
    components: NDArray
    variances: NDArray


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


def fit_marginal_mahalanobis(z_b: NDArray, ridge: float) -> MarginalMahalanobisModel:
    if len(z_b) < 2:
        raise ValueError("Need at least two benign embeddings to fit Mahalanobis distance")
    if ridge < 0:
        raise ValueError("--maha-ridge must be non-negative")

    mean = np.mean(z_b, axis=0)
    centered = z_b - mean
    covariance = (centered.T @ centered) / len(z_b)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    variances = np.maximum(eigvals, 0.0) + ridge
    return MarginalMahalanobisModel(
        mean=mean.astype(np.float64),
        components=eigvecs.astype(np.float64),
        variances=variances.astype(np.float64),
    )


def mahalanobis_distances(model: MarginalMahalanobisModel, z: NDArray) -> NDArray:
    centered = np.asarray(z, dtype=np.float64) - model.mean
    projected = centered @ model.components
    return np.sum((projected * projected) / model.variances, axis=1)


def aggregate_distances(distances: NDArray, aggregation: Aggregation) -> float:
    if aggregation == "mean":
        return float(np.mean(distances))
    if aggregation == "median":
        return float(np.median(distances))
    if aggregation == "p90":
        return float(np.percentile(distances, 90.0))
    if aggregation == "max":
        return float(np.max(distances))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def score_embeddings(model: MarginalMahalanobisModel, z: NDArray, aggregation: Aggregation) -> float:
    return aggregate_distances(mahalanobis_distances(model, z), aggregation)


def sample_batch(z: NDArray, size: int, rng: object) -> NDArray:
    replace = len(z) < size
    idx = rng.choice(len(z), size=size, replace=replace)
    return z[idx]


def build_null_distribution(
    model: MarginalMahalanobisModel,
    z_calibration: NDArray,
    batch_size: int,
    null_samples: int,
    aggregation: Aggregation,
    rng: object,
) -> NDArray:
    scores = np.empty(null_samples, dtype=np.float64)
    for i in range(null_samples):
        batch = sample_batch(z_calibration, batch_size, rng)
        scores[i] = score_embeddings(model, batch, aggregation)
    return scores


def iter_batches(z: NDArray, batch_size: int, drop_last: bool):
    for start in range(0, len(z), batch_size):
        batch = z[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        if len(batch) >= 1:
            yield batch


def score_batch(
    model: MarginalMahalanobisModel,
    z_t: NDArray,
    aggregation: Aggregation,
    null_scores: NDArray,
    threshold: float,
) -> tuple[float, float, bool]:
    score = score_embeddings(model, z_t, aggregation)
    p_value = float((np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0))
    return score, p_value, bool(score > threshold)


def evaluate_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    model: MarginalMahalanobisModel,
    batch_size: int,
    drop_last: bool,
    aggregation: Aggregation,
    null_scores: NDArray,
    threshold: float,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for batch_id, batch in enumerate(iter_batches(z_stream, batch_size, drop_last)):
        score, p_value, suspicious = score_batch(model, batch, aggregation, null_scores, threshold)
        results.append(
            BatchScore(
                split=split,
                batch_id=batch_id,
                label=label,
                size=len(batch),
                attacker_ratio=float(label),
                attacker_count=len(batch) if label == 1 else 0,
                score=score,
                p_value=p_value,
                suspicious=suspicious,
            )
        )
    return results


def evaluate_random_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    model: MarginalMahalanobisModel,
    batch_size: int,
    num_batches: int,
    aggregation: Aggregation,
    null_scores: NDArray,
    threshold: float,
    rng: object,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for batch_id in range(num_batches):
        batch = sample_batch(z_stream, batch_size, rng)
        score, p_value, suspicious = score_batch(model, batch, aggregation, null_scores, threshold)
        results.append(
            BatchScore(
                split=split,
                batch_id=batch_id,
                label=label,
                size=len(batch),
                attacker_ratio=float(label),
                attacker_count=len(batch) if label == 1 else 0,
                score=score,
                p_value=p_value,
                suspicious=suspicious,
            )
        )
    return results


def evaluate_mixed_stream(
    z_benign_stream: NDArray,
    z_attacker_stream: NDArray,
    model: MarginalMahalanobisModel,
    batch_size: int,
    attacker_ratios: list[float],
    batches_per_ratio: int,
    aggregation: Aggregation,
    null_scores: NDArray,
    threshold: float,
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
            score, p_value, suspicious = score_batch(model, batch, aggregation, null_scores, threshold)
            results.append(
                BatchScore(
                    split=f"mixed_{requested_ratio:g}",
                    batch_id=batch_id,
                    label=1,
                    size=len(batch),
                    attacker_ratio=actual_ratio,
                    attacker_count=attacker_count,
                    score=score,
                    p_value=p_value,
                    suspicious=suspicious,
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
                "p_value",
                "suspicious",
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
                    "p_value": r.p_value,
                    "suspicious": r.suspicious,
                }
            )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect attacker query batches using marginal Mahalanobis distance."
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
    parser.add_argument("--device", default=None, help="SentenceTransformer device, e.g. cuda or cpu")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--null-samples", type=int, default=1000)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--normal-train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--normal-calibration-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of normal queries used only to calibrate the Mahalanobis "
            "null distribution. These queries are not used to fit covariance."
        ),
    )
    parser.add_argument(
        "--maha-ridge",
        type=float,
        default=1e-6,
        help="Non-negative diagonal/eigenvalue regularizer for covariance inversion.",
    )
    parser.add_argument(
        "--maha-aggregation",
        choices=["mean", "median", "p90", "max"],
        default="mean",
        help="How to aggregate per-query Mahalanobis distances into a batch score.",
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
    parser.add_argument("--cache-dir", type=Path, default=Path("Mahalanobis/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("Mahalanobis/outputs"))
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

    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")
    if not 0.0 < args.normal_calibration_ratio < 1.0:
        raise ValueError("--normal-calibration-ratio must be between 0 and 1")
    if args.normal_train_ratio + args.normal_calibration_ratio >= 1.0:
        raise ValueError(
            "--normal-train-ratio + --normal-calibration-ratio must be less than 1"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    resolved_dataset = args.dataset
    if args.normal_path is None and args.attacker_path is None and args.normal_source == "dataset":
        normal_path, attacker_path, resolved_dataset = default_dataset_paths(args.data_root, args.dataset)
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

    fit_at = int(len(normal_queries) * args.normal_train_ratio)
    calibration_at = fit_at + int(len(normal_queries) * args.normal_calibration_ratio)
    benign_fit_queries = normal_queries[:fit_at]
    benign_calibration_queries = normal_queries[fit_at:calibration_at]
    benign_test_queries = normal_queries[calibration_at:]
    if len(benign_fit_queries) < 2:
        raise ValueError("Benign fit pool has fewer than 2 queries")
    if len(benign_calibration_queries) < 1:
        raise ValueError("Benign calibration pool has fewer than 1 query")
    if len(benign_test_queries) < 1:
        raise ValueError("Held-out benign test stream has fewer than 1 query")

    all_queries = benign_fit_queries + benign_calibration_queries + benign_test_queries + attacker_queries
    embeddings = embed_texts(
        all_queries,
        model_name=args.embedding_model,
        batch_size=args.encode_batch_size,
        device=args.device,
        prefix=args.query_prefix,
        normalize=not args.no_normalize_embeddings,
        cache_dir=args.cache_dir,
    )

    n_b = len(benign_fit_queries)
    n_c = len(benign_calibration_queries)
    n_h = len(benign_test_queries)
    z_b = embeddings[:n_b]
    z_benign_calibration = embeddings[n_b : n_b + n_c]
    z_benign_test = embeddings[n_b + n_c : n_b + n_c + n_h]
    z_attacker = embeddings[n_b + n_c + n_h :]

    model = fit_marginal_mahalanobis(z_b, args.maha_ridge)
    aggregation = args.maha_aggregation
    null_scores = build_null_distribution(
        model=model,
        z_calibration=z_benign_calibration,
        batch_size=args.batch_size,
        null_samples=args.null_samples,
        aggregation=aggregation,
        rng=rng,
    )
    threshold = float(np.percentile(null_scores, args.threshold_percentile))

    results = []
    if args.benign_eval_batches > 0:
        results.extend(
            evaluate_random_stream(
                split="heldout_benign",
                label=0,
                z_stream=z_benign_test,
                model=model,
                batch_size=args.batch_size,
                num_batches=args.benign_eval_batches,
                aggregation=aggregation,
                null_scores=null_scores,
                threshold=threshold,
                rng=rng,
            )
        )
    else:
        results.extend(
            evaluate_stream(
                split="heldout_benign",
                label=0,
                z_stream=z_benign_test,
                model=model,
                batch_size=args.batch_size,
                drop_last=not args.keep_last_batch,
                aggregation=aggregation,
                null_scores=null_scores,
                threshold=threshold,
            )
        )
    if args.attacker_eval_batches > 0:
        results.extend(
            evaluate_random_stream(
                split="attacker",
                label=1,
                z_stream=z_attacker,
                model=model,
                batch_size=args.batch_size,
                num_batches=args.attacker_eval_batches,
                aggregation=aggregation,
                null_scores=null_scores,
                threshold=threshold,
                rng=rng,
            )
        )
    else:
        results.extend(
            evaluate_stream(
                split="attacker",
                label=1,
                z_stream=z_attacker,
                model=model,
                batch_size=args.batch_size,
                drop_last=not args.keep_last_batch,
                aggregation=aggregation,
                null_scores=null_scores,
                threshold=threshold,
            )
        )
    if mixed_ratios:
        results.extend(
            evaluate_mixed_stream(
                z_benign_stream=z_benign_test,
                z_attacker_stream=z_attacker,
                model=model,
                batch_size=args.batch_size,
                attacker_ratios=mixed_ratios,
                batches_per_ratio=args.mixed_batches_per_ratio,
                aggregation=aggregation,
                null_scores=null_scores,
                threshold=threshold,
                rng=rng,
            )
        )

    metrics = summarize(results)
    metrics_by_split = summarize_by_split(results)
    metadata = {
        "detector": "marginal_mahalanobis",
        "dataset": resolved_dataset,
        "data_root": str(args.data_root),
        "normal_source": args.normal_source,
        "global_normal_path": str(args.global_normal_path),
        "normal_path": str(normal_path),
        "attacker_path": str(attacker_path),
        "text_field": args.text_field,
        "embedding_model": args.embedding_model,
        "query_prefix": args.query_prefix,
        "normalize_embeddings": not args.no_normalize_embeddings,
        "batch_size": args.batch_size,
        "null_samples": args.null_samples,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "normal_calibration_ratio": args.normal_calibration_ratio,
        "maha_ridge": args.maha_ridge,
        "maha_aggregation": aggregation,
        "normal_queries": len(normal_queries),
        "benign_pool_size": len(benign_fit_queries),
        "benign_fit_size": len(benign_fit_queries),
        "benign_calibration_size": len(benign_calibration_queries),
        "heldout_benign_size": len(benign_test_queries),
        "attacker_queries": len(attacker_queries),
        "embedding_dim": int(z_b.shape[1]),
        "covariance_rank_estimate": int(np.sum(model.variances > args.maha_ridge)),
        "min_covariance_variance": float(np.min(model.variances)),
        "max_covariance_variance": float(np.max(model.variances)),
        "null_score_mean": float(np.mean(null_scores)),
        "null_score_std": float(np.std(null_scores)),
        "null_score_min": float(np.min(null_scores)),
        "null_score_max": float(np.max(null_scores)),
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
