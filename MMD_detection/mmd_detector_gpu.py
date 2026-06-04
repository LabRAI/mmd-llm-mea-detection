#!/usr/bin/env python3
"""Batch-level attacker detection with unbiased MMD on query embeddings using PyTorch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

np = None
torch = None

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
    p_value: float
    suspicious: bool


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


def squared_distances(x: NDArray, y: NDArray) -> NDArray:
    x_norm = torch.sum(x * x, dim=1, keepdim=True)
    y_norm = torch.sum(y * y, dim=1, keepdim=True).T
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return torch.clamp(d2, min=0.0)


def median_bandwidth(z: NDArray, sample_size: int, rng: object) -> float:
    n = len(z)
    if n < 2:
        raise ValueError("Need at least two benign embeddings to estimate bandwidth")
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=z.device)
    d2 = squared_distances(z[idx_t], z[idx_t])
    tri = torch.triu_indices(len(idx), len(idx), offset=1, device=z.device)
    upper = d2[tri[0], tri[1]]
    positive = upper[upper > 0]
    if positive.numel() == 0:
        return 1.0
    return float(torch.sqrt(torch.median(positive)).detach().cpu().item())


def rbf_kernel_from_d2(d2: NDArray, sigmas: Iterable[float]) -> NDArray:
    kernels = []
    for sigma in sigmas:
        sigma = max(float(sigma), 1e-12)
        kernels.append(torch.exp(-d2 / (2.0 * sigma * sigma)))
    return torch.stack(kernels, dim=0).mean(dim=0)


def mmd2_unbiased(x: NDArray, y: NDArray, sigmas: list[float]) -> float:
    m, n = len(x), len(y)
    if m < 2 or n < 2:
        raise ValueError("Unbiased MMD requires both batches to contain at least 2 samples")

    k_xx = rbf_kernel_from_d2(squared_distances(x, x), sigmas)
    k_yy = rbf_kernel_from_d2(squared_distances(y, y), sigmas)
    k_xy = rbf_kernel_from_d2(squared_distances(x, y), sigmas)

    xx = (torch.sum(k_xx) - torch.trace(k_xx)) / (m * (m - 1))
    yy = (torch.sum(k_yy) - torch.trace(k_yy)) / (n * (n - 1))
    xy = torch.sum(k_xy) / (m * n)
    return float((xx + yy - 2.0 * xy).detach().cpu().item())


def sample_batch(z: NDArray, size: int, rng: object) -> NDArray:
    replace = len(z) < size
    idx = rng.choice(len(z), size=size, replace=replace)
    if torch is not None and torch.is_tensor(z):
        idx = torch.as_tensor(idx, dtype=torch.long, device=z.device)
    return z[idx]


def build_null_distribution(
    z_b: NDArray,
    batch_size: int,
    null_samples: int,
    sigmas: list[float],
    rng: object,
) -> NDArray:
    scores = np.empty(null_samples, dtype=np.float64)
    can_disjoint = len(z_b) >= 2 * batch_size
    for i in range(null_samples):
        if can_disjoint:
            idx = rng.choice(len(z_b), size=2 * batch_size, replace=False)
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=z_b.device)
            r1 = z_b[idx_t[:batch_size]]
            r2 = z_b[idx_t[batch_size:]]
        else:
            r1 = sample_batch(z_b, batch_size, rng)
            r2 = sample_batch(z_b, batch_size, rng)
        scores[i] = mmd2_unbiased(r1, r2, sigmas)
    return scores


def iter_batches(z: NDArray, batch_size: int, drop_last: bool) -> Iterable[NDArray]:
    for start in range(0, len(z), batch_size):
        batch = z[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        if len(batch) >= 2:
            yield batch


def score_batch(
    z_b: NDArray,
    z_t: NDArray,
    reference_repeats: int,
    sigmas: list[float],
    null_scores: NDArray,
    threshold: float,
    rng: object,
) -> tuple[float, float, bool]:
    scores = []
    for _ in range(reference_repeats):
        ref = sample_batch(z_b, len(z_t), rng)
        scores.append(mmd2_unbiased(ref, z_t, sigmas))
    score = float(np.mean(scores))
    p_value = float((np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0))
    return score, p_value, bool(score > threshold)


def evaluate_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    z_b: NDArray,
    batch_size: int,
    drop_last: bool,
    reference_repeats: int,
    sigmas: list[float],
    null_scores: NDArray,
    threshold: float,
    rng: object,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for batch_id, batch in enumerate(iter_batches(z_stream, batch_size, drop_last)):
        score, p_value, suspicious = score_batch(
            z_b, batch, reference_repeats, sigmas, null_scores, threshold, rng
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
                p_value=p_value,
                suspicious=suspicious,
            )
        )
    return results


def evaluate_random_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    z_b: NDArray,
    batch_size: int,
    num_batches: int,
    reference_repeats: int,
    sigmas: list[float],
    null_scores: NDArray,
    threshold: float,
    rng: object,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for batch_id in range(num_batches):
        batch = sample_batch(z_stream, batch_size, rng)
        score, p_value, suspicious = score_batch(
            z_b, batch, reference_repeats, sigmas, null_scores, threshold, rng
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
                p_value=p_value,
                suspicious=suspicious,
            )
        )
    return results


def evaluate_mixed_stream(
    z_benign_stream: NDArray,
    z_attacker_stream: NDArray,
    z_b: NDArray,
    batch_size: int,
    attacker_ratios: list[float],
    batches_per_ratio: int,
    reference_repeats: int,
    sigmas: list[float],
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
            batch = torch.cat([benign_part, attacker_part], dim=0)
            perm = torch.as_tensor(rng.permutation(len(batch)), dtype=torch.long, device=batch.device)
            batch = batch[perm]
            score, p_value, suspicious = score_batch(
                z_b, batch, reference_repeats, sigmas, null_scores, threshold, rng
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
        description="Detect attacker query batches using unbiased MMD over embeddings."
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
    parser.add_argument(
        "--mmd-device",
        default=None,
        help="PyTorch device for MMD computation. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--mmd-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Tensor dtype for GPU MMD computation.",
    )
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--reference-repeats", type=int, default=20)
    parser.add_argument("--null-samples", type=int, default=1000)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--normal-train-ratio", type=float, default=0.8)
    parser.add_argument("--bandwidth-sample-size", type=int, default=2000)
    parser.add_argument(
        "--multi-kernel",
        action="store_true",
        help="Use [0.5, 1, 2, 4] times the median bandwidth and average kernels.",
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
    parser.add_argument("--cache-dir", type=Path, default=Path("MMD_detection/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("MMD_detection/outputs"))
    return parser.parse_args()


def main() -> None:
    global np, torch
    args = parse_args()
    try:
        import numpy as _np
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: numpy. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    np = _np
    try:
        import torch as _torch
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: torch. Install PyTorch to use mmd_detector_gpu.py."
        ) from exc
    torch = _torch

    if args.mmd_device is None:
        mmd_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        mmd_device = args.mmd_device
    if str(mmd_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested --mmd-device {mmd_device}, but torch.cuda.is_available() is False"
        )
    mmd_dtype = torch.float64 if args.mmd_dtype == "float64" else torch.float32

    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")

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
    if len(benign_pool_queries) < args.batch_size:
        raise ValueError("Benign reference pool is smaller than --batch-size")
    if len(benign_test_queries) < 2:
        raise ValueError("Held-out benign test stream has fewer than 2 queries")

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
    embeddings_t = torch.as_tensor(embeddings, dtype=mmd_dtype, device=mmd_device)
    z_b = embeddings_t[:n_b]
    z_benign_test = embeddings_t[n_b : n_b + n_h]
    z_attacker = embeddings_t[n_b + n_h :]

    sigma = median_bandwidth(z_b, args.bandwidth_sample_size, rng)
    sigmas = [0.5 * sigma, sigma, 2.0 * sigma, 4.0 * sigma] if args.multi_kernel else [sigma]

    null_scores = build_null_distribution(
        z_b=z_b,
        batch_size=args.batch_size,
        null_samples=args.null_samples,
        sigmas=sigmas,
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
                z_b=z_b,
                batch_size=args.batch_size,
                num_batches=args.benign_eval_batches,
                reference_repeats=args.reference_repeats,
                sigmas=sigmas,
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
                z_b=z_b,
                batch_size=args.batch_size,
                drop_last=not args.keep_last_batch,
                reference_repeats=args.reference_repeats,
                sigmas=sigmas,
                null_scores=null_scores,
                threshold=threshold,
                rng=rng,
            )
        )
    if args.attacker_eval_batches > 0:
        results.extend(
            evaluate_random_stream(
                split="attacker",
                label=1,
                z_stream=z_attacker,
                z_b=z_b,
                batch_size=args.batch_size,
                num_batches=args.attacker_eval_batches,
                reference_repeats=args.reference_repeats,
                sigmas=sigmas,
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
                z_b=z_b,
                batch_size=args.batch_size,
                drop_last=not args.keep_last_batch,
                reference_repeats=args.reference_repeats,
                sigmas=sigmas,
                null_scores=null_scores,
                threshold=threshold,
                rng=rng,
            )
        )
    if mixed_ratios:
        results.extend(
            evaluate_mixed_stream(
                z_benign_stream=z_benign_test,
                z_attacker_stream=z_attacker,
                z_b=z_b,
                batch_size=args.batch_size,
                attacker_ratios=mixed_ratios,
                batches_per_ratio=args.mixed_batches_per_ratio,
                reference_repeats=args.reference_repeats,
                sigmas=sigmas,
                null_scores=null_scores,
                threshold=threshold,
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
        "mmd_backend": "torch",
        "mmd_device": str(embeddings_t.device),
        "mmd_dtype": args.mmd_dtype,
        "query_prefix": args.query_prefix,
        "normalize_embeddings": not args.no_normalize_embeddings,
        "batch_size": args.batch_size,
        "reference_repeats": args.reference_repeats,
        "null_samples": args.null_samples,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "median_bandwidth": sigma,
        "sigmas": sigmas,
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
