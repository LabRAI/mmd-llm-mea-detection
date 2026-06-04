#!/usr/bin/env python3
"""Batch-level SEAT-style attacker detection on LLM query embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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
    p_value: float
    suspicious: bool
    similar_pair_count: int
    total_pair_count: int


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


def l2_normalize(z: NDArray) -> NDArray:
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norms, 1e-12)


def move_to_compute_device(z: NDArray, compute_device: str | None):
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


def to_numpy_1d(x):
    if is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return x


def take_rows(z: NDArray, idx: NDArray):
    if is_torch_tensor(z):
        import torch

        return z[torch.as_tensor(idx, dtype=torch.long, device=z.device)]
    return z[idx]


def concat_rows(parts: list[NDArray]):
    if parts and is_torch_tensor(parts[0]):
        import torch

        return torch.cat(parts, dim=0)
    return np.concatenate(parts, axis=0)


def permute_rows(z: NDArray, order: NDArray):
    return take_rows(z, order)


def upper_triangle_values(matrix: NDArray) -> NDArray:
    if is_torch_tensor(matrix):
        import torch

        idx = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1, device=matrix.device)
        return matrix[idx[0], idx[1]]
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def pairwise_cosine_values(z: NDArray) -> NDArray:
    if len(z) < 2:
        return np.asarray([], dtype=np.float32)
    sim = z @ z.T
    return upper_triangle_values(sim)


def score_similar_pairs(
    z_batch: NDArray,
    similarity_threshold: float,
    score_mode: str,
) -> tuple[float, int, int]:
    sims = pairwise_cosine_values(z_batch)
    total_pairs = int(len(sims))
    if total_pairs == 0:
        raise ValueError("Need at least two samples to score a batch")
    if is_torch_tensor(sims):
        similar_pair_count = int((sims >= similarity_threshold).sum().item())
    else:
        similar_pair_count = int(np.sum(sims >= similarity_threshold))
    if score_mode == "count":
        score = float(similar_pair_count)
    elif score_mode == "ratio":
        score = float(similar_pair_count / total_pairs)
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")
    return score, similar_pair_count, total_pairs


def sample_batch(z: NDArray, size: int, rng: object) -> NDArray:
    replace = len(z) < size
    idx = rng.choice(len(z), size=size, replace=replace)
    return take_rows(z, idx)


def sample_pair_similarities(z: NDArray, sample_size: int, rng: object) -> NDArray:
    if len(z) < 2:
        raise ValueError("Need at least two benign embeddings to estimate pair similarity threshold")
    if len(z) <= 5000:
        sims = pairwise_cosine_values(z)
        sims = to_numpy_1d(sims)
        if len(sims) <= sample_size:
            return sims
        idx = rng.choice(len(sims), size=sample_size, replace=False)
        return sims[idx]

    first = rng.integers(0, len(z), size=sample_size)
    second = rng.integers(0, len(z) - 1, size=sample_size)
    second = np.where(second >= first, second + 1, second)
    first_rows = take_rows(z, first)
    second_rows = take_rows(z, second)
    if is_torch_tensor(z):
        return to_numpy_1d((first_rows * second_rows).sum(dim=1))
    return np.sum(first_rows * second_rows, axis=1)


def calibrate_similarity_threshold(
    z_b: NDArray,
    percentile: float,
    sample_size: int,
    rng: object,
) -> float:
    pair_sims = sample_pair_similarities(z_b, sample_size, rng)
    if len(pair_sims) == 0:
        raise ValueError("No benign pair similarities were available for calibration")
    return float(np.percentile(pair_sims, percentile))


def build_null_distribution(
    z_b: NDArray,
    batch_size: int,
    null_samples: int,
    similarity_threshold: float,
    score_mode: str,
    rng: object,
) -> NDArray:
    scores = np.empty(null_samples, dtype=np.float64)
    for i in range(null_samples):
        batch = sample_batch(z_b, batch_size, rng)
        scores[i], _, _ = score_similar_pairs(batch, similarity_threshold, score_mode)
    return scores


def iter_batches(z: NDArray, batch_size: int, drop_last: bool) -> Iterable[NDArray]:
    for start in range(0, len(z), batch_size):
        batch = z[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        if len(batch) >= 2:
            yield batch


def score_batch(
    z_t: NDArray,
    similarity_threshold: float,
    score_mode: str,
    null_scores: NDArray,
    lower_threshold: float,
    upper_threshold: float,
    detection_tail: str,
) -> tuple[float, float, bool, int, int]:
    score, similar_pair_count, total_pair_count = score_similar_pairs(
        z_t, similarity_threshold, score_mode
    )
    if detection_tail == "upper":
        p_value = float((np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0))
        suspicious = score > upper_threshold
    elif detection_tail == "lower":
        p_value = float((np.sum(null_scores <= score) + 1.0) / (len(null_scores) + 1.0))
        suspicious = score < lower_threshold
    elif detection_tail == "two-sided":
        lower_p = float((np.sum(null_scores <= score) + 1.0) / (len(null_scores) + 1.0))
        upper_p = float((np.sum(null_scores >= score) + 1.0) / (len(null_scores) + 1.0))
        p_value = min(1.0, 2.0 * min(lower_p, upper_p))
        suspicious = score < lower_threshold or score > upper_threshold
    else:
        raise ValueError(f"Unsupported detection tail: {detection_tail}")
    return score, p_value, bool(suspicious), similar_pair_count, total_pair_count


def evaluate_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    batch_size: int,
    drop_last: bool,
    similarity_threshold: float,
    score_mode: str,
    null_scores: NDArray,
    lower_threshold: float,
    upper_threshold: float,
    detection_tail: str,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for batch_id, batch in enumerate(iter_batches(z_stream, batch_size, drop_last)):
        score, p_value, suspicious, similar_pair_count, total_pair_count = score_batch(
            batch,
            similarity_threshold,
            score_mode,
            null_scores,
            lower_threshold,
            upper_threshold,
            detection_tail,
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
                similar_pair_count=similar_pair_count,
                total_pair_count=total_pair_count,
            )
        )
    return results


def evaluate_random_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    batch_size: int,
    num_batches: int,
    similarity_threshold: float,
    score_mode: str,
    null_scores: NDArray,
    lower_threshold: float,
    upper_threshold: float,
    detection_tail: str,
    rng: object,
) -> list[BatchScore]:
    results: list[BatchScore] = []
    for batch_id in range(num_batches):
        batch = sample_batch(z_stream, batch_size, rng)
        score, p_value, suspicious, similar_pair_count, total_pair_count = score_batch(
            batch,
            similarity_threshold,
            score_mode,
            null_scores,
            lower_threshold,
            upper_threshold,
            detection_tail,
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
                similar_pair_count=similar_pair_count,
                total_pair_count=total_pair_count,
            )
        )
    return results


def evaluate_mixed_stream(
    z_benign_stream: NDArray,
    z_attacker_stream: NDArray,
    batch_size: int,
    attacker_ratios: list[float],
    batches_per_ratio: int,
    similarity_threshold: float,
    score_mode: str,
    null_scores: NDArray,
    lower_threshold: float,
    upper_threshold: float,
    detection_tail: str,
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
            batch = concat_rows([benign_part, attacker_part])
            batch = permute_rows(batch, rng.permutation(len(batch)))
            score, p_value, suspicious, similar_pair_count, total_pair_count = score_batch(
                batch,
                similarity_threshold,
                score_mode,
                null_scores,
                lower_threshold,
                upper_threshold,
                detection_tail,
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
                    similar_pair_count=similar_pair_count,
                    total_pair_count=total_pair_count,
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
            "mean_similar_pair_count": (
                float(np.mean([r.similar_pair_count for r in split_results]))
                if split_results
                else float("nan")
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
                "p_value",
                "suspicious",
                "similar_pair_count",
                "total_pair_count",
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
                    "similar_pair_count": r.similar_pair_count,
                    "total_pair_count": r.total_pair_count,
                }
            )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect attacker query batches with SEAT-style similar-pair counting."
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
        "--compute-device",
        default=None,
        help="Device for pairwise similarity scoring. Defaults to --device when set.",
    )
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--score-mode",
        choices=["ratio", "count"],
        default="ratio",
        help="Use the ratio or raw count of similar pairs as the batch score.",
    )
    parser.add_argument(
        "--detection-tail",
        choices=["upper", "lower", "two-sided"],
        default="two-sided",
        help=(
            "Detect batches with too many similar pairs, too few similar pairs, "
            "or either deviation. upper matches the original SEAT assumption."
        ),
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Cosine threshold delta for a similar pair. If omitted, calibrate from benign pairs.",
    )
    parser.add_argument(
        "--similarity-threshold-percentile",
        type=float,
        default=99.0,
        help="Benign pair cosine percentile used to calibrate delta when --similarity-threshold is omitted.",
    )
    parser.add_argument(
        "--pair-sample-size",
        type=int,
        default=200000,
        help="Number of benign pairs sampled to calibrate the similarity threshold.",
    )
    parser.add_argument("--null-samples", type=int, default=1000)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--normal-train-ratio", type=float, default=0.8)
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
    parser.add_argument("--cache-dir", type=Path, default=Path("SEAT/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("SEAT/outputs"))
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
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.null_samples < 1:
        raise ValueError("--null-samples must be at least 1")
    if not 0.0 <= args.threshold_percentile <= 100.0:
        raise ValueError("--threshold-percentile must be between 0 and 100")
    if not 0.0 <= args.similarity_threshold_percentile <= 100.0:
        raise ValueError("--similarity-threshold-percentile must be between 0 and 100")

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
    embeddings = l2_normalize(embeddings)
    embeddings, compute_backend = move_to_compute_device(embeddings, args.compute_device or args.device)

    n_b = len(benign_pool_queries)
    n_h = len(benign_test_queries)
    z_b = embeddings[:n_b]
    z_benign_test = embeddings[n_b : n_b + n_h]
    z_attacker = embeddings[n_b + n_h :]

    if args.similarity_threshold is None:
        similarity_threshold = calibrate_similarity_threshold(
            z_b=z_b,
            percentile=args.similarity_threshold_percentile,
            sample_size=args.pair_sample_size,
            rng=rng,
        )
    else:
        similarity_threshold = float(args.similarity_threshold)

    null_scores = build_null_distribution(
        z_b=z_b,
        batch_size=args.batch_size,
        null_samples=args.null_samples,
        similarity_threshold=similarity_threshold,
        score_mode=args.score_mode,
        rng=rng,
    )
    upper_threshold = float(np.percentile(null_scores, args.threshold_percentile))
    lower_threshold = float(np.percentile(null_scores, 100.0 - args.threshold_percentile))

    results = []
    if args.benign_eval_batches > 0:
        results.extend(
            evaluate_random_stream(
                split="heldout_benign",
                label=0,
                z_stream=z_benign_test,
                batch_size=args.batch_size,
                num_batches=args.benign_eval_batches,
                similarity_threshold=similarity_threshold,
                score_mode=args.score_mode,
                null_scores=null_scores,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                detection_tail=args.detection_tail,
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
                similarity_threshold=similarity_threshold,
                score_mode=args.score_mode,
                null_scores=null_scores,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                detection_tail=args.detection_tail,
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
                similarity_threshold=similarity_threshold,
                score_mode=args.score_mode,
                null_scores=null_scores,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                detection_tail=args.detection_tail,
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
                similarity_threshold=similarity_threshold,
                score_mode=args.score_mode,
                null_scores=null_scores,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                detection_tail=args.detection_tail,
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
                similarity_threshold=similarity_threshold,
                score_mode=args.score_mode,
                null_scores=null_scores,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                detection_tail=args.detection_tail,
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
        "batch_size": args.batch_size,
        "score_mode": args.score_mode,
        "detection_tail": args.detection_tail,
        "similarity_threshold": similarity_threshold,
        "similarity_threshold_source": (
            "manual" if args.similarity_threshold is not None else "benign_pair_percentile"
        ),
        "similarity_threshold_percentile": args.similarity_threshold_percentile,
        "pair_sample_size": args.pair_sample_size,
        "null_samples": args.null_samples,
        "threshold_percentile": args.threshold_percentile,
        "threshold": upper_threshold,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
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
