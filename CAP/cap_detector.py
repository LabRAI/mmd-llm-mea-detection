#!/usr/bin/env python3
"""Coverage-Aware Perturbation detector for attacker query streams.

This implements the detection part of CAP from "Stealing and Defending the
Ends of LLMs": track how widely a user's queries cover embedding space, map
coverage metrics to a cost, and flag users whose cost is above a benign
calibration threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MMD_detection import mmd_detector as base

np = None

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = object


@dataclass
class CAPBatchScore:
    split: str
    batch_id: int
    label: int
    size: int
    attacker_ratio: float
    attacker_count: int
    total_queries: int
    bucket_coverage: float
    new_bucket_rate: float
    spread: float
    cost: float
    low_threshold: float
    high_threshold: float
    suspicious: bool


@dataclass
class CAPConfig:
    num_buckets: int = 2048
    baseline_lambda: float = 0.0005
    alpha: float = 8.0
    beta: float = 0.19
    coverage_weight: float = 0.05
    novelty_weight: float = 0.35
    spread_weight: float = 0.45
    spread_max: float = 1.0
    max_cost: float = 1e12


class CAPDetector:
    """Stateful per-user CAP coverage detector."""

    def __init__(
        self,
        embedding_dim: int,
        config: CAPConfig,
        rng: object,
        random_planes: NDArray | None = None,
    ) -> None:
        if config.num_buckets < 2:
            raise ValueError("num_buckets must be at least 2")
        if config.baseline_lambda <= 0.0:
            raise ValueError("baseline_lambda must be positive")
        if config.alpha <= 0.0 or config.beta <= 0.0:
            raise ValueError("alpha and beta must be positive")
        if config.spread_max <= 0.0:
            raise ValueError("spread_max must be positive")

        self.config = config
        self.embedding_dim = embedding_dim
        self.num_bits = int(math.ceil(math.log2(config.num_buckets)))
        self.random_planes = (
            random_planes
            if random_planes is not None
            else rng.normal(size=(embedding_dim, self.num_bits)).astype(np.float32)
        )
        self.seen_buckets: set[int] = set()
        self.total_queries = 0
        self.sum_embedding = np.zeros(embedding_dim, dtype=np.float64)
        self.sum_squared_norm = 0.0

    def hash_embeddings(self, embeddings: NDArray) -> NDArray:
        projections = embeddings @ self.random_planes
        bits = projections >= 0.0
        powers = (1 << np.arange(self.num_bits, dtype=np.uint64))
        bucket_ids = (bits.astype(np.uint64) @ powers) % self.config.num_buckets
        return bucket_ids.astype(np.int64)

    def update(self, embeddings: NDArray) -> tuple[float, float, float, float, int]:
        if len(embeddings) == 0:
            raise ValueError("CAPDetector.update requires a non-empty batch")

        batch = np.asarray(embeddings, dtype=np.float32)
        bucket_ids = self.hash_embeddings(batch)
        old_bucket_count = len(self.seen_buckets)
        self.seen_buckets.update(int(bucket_id) for bucket_id in bucket_ids)
        new_bucket_count = len(self.seen_buckets) - old_bucket_count

        self.total_queries += len(batch)
        self.sum_embedding += np.sum(batch, axis=0, dtype=np.float64)
        self.sum_squared_norm += float(np.sum(batch.astype(np.float64) * batch.astype(np.float64)))

        coverage = len(self.seen_buckets) / self.config.num_buckets
        new_bucket_rate = new_bucket_count / len(batch)
        spread = self.current_spread()
        cost = self.cost(coverage, new_bucket_rate, spread)
        return coverage, new_bucket_rate, spread, cost, self.total_queries

    def current_spread(self) -> float:
        if self.total_queries <= 1:
            return 0.0
        mean = self.sum_embedding / self.total_queries
        mean_squared_distance = self.sum_squared_norm / self.total_queries - float(mean @ mean)
        return float(math.sqrt(max(mean_squared_distance, 0.0)))

    def cost(self, coverage: float, new_bucket_rate: float, spread: float) -> float:
        cfg = self.config
        exponent = (coverage / cfg.beta) * math.log(cfg.alpha / cfg.baseline_lambda)
        coverage_penalty = cfg.coverage_weight * (math.exp(min(exponent, 700.0)) - 1.0)
        novelty_penalty = cfg.alpha * cfg.novelty_weight * new_bucket_rate
        spread_penalty = cfg.alpha * cfg.spread_weight * min(spread / cfg.spread_max, 1.0)
        total = cfg.baseline_lambda + coverage_penalty + novelty_penalty + spread_penalty
        return min(float(total), cfg.max_cost)


def iter_batches(z: NDArray, batch_size: int, drop_last: bool) -> Iterable[NDArray]:
    for start in range(0, len(z), batch_size):
        batch = z[start : start + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        if len(batch) > 0:
            yield batch


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_detector(embedding_dim: int, config: CAPConfig, random_planes: NDArray) -> CAPDetector:
    return CAPDetector(
        embedding_dim=embedding_dim,
        config=config,
        rng=np.random.default_rng(0),
        random_planes=random_planes,
    )


def score_stream(
    split: str,
    label: int,
    z_stream: NDArray,
    batch_size: int,
    drop_last: bool,
    config: CAPConfig,
    random_planes: NDArray,
    low_threshold: float,
    high_threshold: float,
    tail: str,
) -> list[CAPBatchScore]:
    detector = make_detector(z_stream.shape[1], config, random_planes)
    results: list[CAPBatchScore] = []
    for batch_id, batch in enumerate(iter_batches(z_stream, batch_size, drop_last)):
        coverage, novelty, spread, cost, total_queries = detector.update(batch)
        suspicious = is_suspicious(cost, low_threshold, high_threshold, tail)
        results.append(
            CAPBatchScore(
                split=split,
                batch_id=batch_id,
                label=label,
                size=len(batch),
                attacker_ratio=float(label),
                attacker_count=len(batch) if label == 1 else 0,
                total_queries=total_queries,
                bucket_coverage=coverage,
                new_bucket_rate=novelty,
                spread=spread,
                cost=cost,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                suspicious=suspicious,
            )
        )
    return results


def score_random_streams(
    split: str,
    label: int,
    z_stream: NDArray,
    batch_size: int,
    num_streams: int,
    stream_length_batches: int,
    config: CAPConfig,
    random_planes: NDArray,
    low_threshold: float,
    high_threshold: float,
    tail: str,
    rng: object,
) -> list[CAPBatchScore]:
    results: list[CAPBatchScore] = []
    for stream_id in range(num_streams):
        detector = make_detector(z_stream.shape[1], config, random_planes)
        for batch_id in range(stream_length_batches):
            batch = base.sample_batch(z_stream, batch_size, rng)
            coverage, novelty, spread, cost, total_queries = detector.update(batch)
            suspicious = is_suspicious(cost, low_threshold, high_threshold, tail)
            results.append(
                CAPBatchScore(
                    split=f"{split}_stream_{stream_id}",
                    batch_id=batch_id,
                    label=label,
                    size=len(batch),
                    attacker_ratio=float(label),
                    attacker_count=len(batch) if label == 1 else 0,
                    total_queries=total_queries,
                    bucket_coverage=coverage,
                    new_bucket_rate=novelty,
                    spread=spread,
                    cost=cost,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    suspicious=suspicious,
                )
            )
    return results


def score_mixed_streams(
    z_benign_stream: NDArray,
    z_attacker_stream: NDArray,
    batch_size: int,
    attacker_ratios: list[float],
    streams_per_ratio: int,
    stream_length_batches: int,
    config: CAPConfig,
    random_planes: NDArray,
    low_threshold: float,
    high_threshold: float,
    tail: str,
    rng: object,
) -> list[CAPBatchScore]:
    results: list[CAPBatchScore] = []
    for requested_ratio in attacker_ratios:
        attacker_count = int(round(batch_size * requested_ratio))
        attacker_count = min(max(attacker_count, 1), batch_size - 1)
        benign_count = batch_size - attacker_count
        actual_ratio = attacker_count / batch_size
        for stream_id in range(streams_per_ratio):
            detector = make_detector(z_benign_stream.shape[1], config, random_planes)
            for batch_id in range(stream_length_batches):
                benign_part = base.sample_batch(z_benign_stream, benign_count, rng)
                attacker_part = base.sample_batch(z_attacker_stream, attacker_count, rng)
                batch = np.concatenate([benign_part, attacker_part], axis=0)
                batch = batch[rng.permutation(len(batch))]
                coverage, novelty, spread, cost, total_queries = detector.update(batch)
                suspicious = is_suspicious(cost, low_threshold, high_threshold, tail)
                results.append(
                    CAPBatchScore(
                        split=f"mixed_{requested_ratio:g}_stream_{stream_id}",
                        batch_id=batch_id,
                        label=1,
                        size=len(batch),
                        attacker_ratio=actual_ratio,
                        attacker_count=attacker_count,
                        total_queries=total_queries,
                        bucket_coverage=coverage,
                        new_bucket_rate=novelty,
                        spread=spread,
                        cost=cost,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                        suspicious=suspicious,
                    )
                )
    return results


def score_global_mixed_stream(
    z_benign_stream: NDArray,
    z_attacker_stream: NDArray,
    batch_size: int,
    attacker_ratios: list[float],
    config: CAPConfig,
    random_planes: NDArray,
    low_threshold: float,
    high_threshold: float,
    tail: str,
    rng: object,
    drop_last: bool,
) -> list[CAPBatchScore]:
    results: list[CAPBatchScore] = []
    for requested_ratio in attacker_ratios:
        attacker_count = int(round(batch_size * requested_ratio))
        attacker_count = min(max(attacker_count, 1), batch_size - 1)
        benign_count = batch_size - attacker_count
        actual_ratio = attacker_count / batch_size
        if drop_last and (len(z_benign_stream) < benign_count or len(z_attacker_stream) < attacker_count):
            continue

        benign_batches = max(1, len(z_benign_stream) // benign_count)
        attacker_batches = max(1, len(z_attacker_stream) // attacker_count)
        max_batches = min(benign_batches, attacker_batches)

        detector = make_detector(z_benign_stream.shape[1], config, random_planes)
        for batch_id in range(max_batches):
            benign_part = base.sample_batch(z_benign_stream, benign_count, rng)
            attacker_part = base.sample_batch(z_attacker_stream, attacker_count, rng)
            if len(benign_part) == 0 or len(attacker_part) == 0:
                continue
            batch = np.concatenate([benign_part, attacker_part], axis=0)
            batch = batch[rng.permutation(len(batch))]
            coverage, novelty, spread, cost, total_queries = detector.update(batch)
            suspicious = is_suspicious(cost, low_threshold, high_threshold, tail)
            results.append(
                CAPBatchScore(
                    split=f"global_mixed_{requested_ratio:g}",
                    batch_id=batch_id,
                    label=1,
                    size=len(batch),
                    attacker_ratio=len(attacker_part) / len(batch),
                    attacker_count=len(attacker_part),
                    total_queries=total_queries,
                    bucket_coverage=coverage,
                    new_bucket_rate=novelty,
                    spread=spread,
                    cost=cost,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    suspicious=suspicious,
                )
            )
    return results


def build_benign_calibration_costs(
    z_calib: NDArray,
    batch_size: int,
    calibration_streams: int,
    stream_length_batches: int,
    config: CAPConfig,
    random_planes: NDArray,
    rng: object,
) -> NDArray:
    costs = []
    for _ in range(calibration_streams):
        detector = make_detector(z_calib.shape[1], config, random_planes)
        for _ in range(stream_length_batches):
            batch = base.sample_batch(z_calib, batch_size, rng)
            _, _, _, cost, _ = detector.update(batch)
            costs.append(cost)
    return np.asarray(costs, dtype=np.float64)


def is_suspicious(cost: float, low_threshold: float, high_threshold: float, tail: str) -> bool:
    if tail == "high":
        return bool(cost > high_threshold)
    if tail == "low":
        return bool(cost < low_threshold)
    if tail == "two-sided":
        return bool(cost < low_threshold or cost > high_threshold)
    raise ValueError(f"Unknown tail: {tail}")


def summarize(results: list[CAPBatchScore]) -> dict[str, float | int]:
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


def summarize_by_family(results: list[CAPBatchScore]) -> dict[str, dict[str, float | int]]:
    families: dict[str, list[CAPBatchScore]] = {}
    for result in results:
        family = result.split.split("_stream_", maxsplit=1)[0]
        families.setdefault(family, []).append(result)

    summaries = {}
    for family, family_results in sorted(families.items()):
        suspicious = sum(r.suspicious for r in family_results)
        summaries[family] = {
            "batches": len(family_results),
            "suspicious": suspicious,
            "detection_rate": suspicious / len(family_results) if family_results else float("nan"),
            "mean_cost": float(np.mean([r.cost for r in family_results])) if family_results else float("nan"),
            "mean_coverage": (
                float(np.mean([r.bucket_coverage for r in family_results]))
                if family_results
                else float("nan")
            ),
            "mean_new_bucket_rate": (
                float(np.mean([r.new_bucket_rate for r in family_results]))
                if family_results
                else float("nan")
            ),
            "mean_spread": (
                float(np.mean([r.spread for r in family_results])) if family_results else float("nan")
            ),
            "mean_attacker_ratio": (
                float(np.mean([r.attacker_ratio for r in family_results]))
                if family_results
                else float("nan")
            ),
        }
    return summaries


def write_outputs(output_dir: Path, results: list[CAPBatchScore], metadata: dict) -> None:
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
                "total_queries",
                "bucket_coverage",
                "new_bucket_rate",
                "spread",
                "cost",
                "low_threshold",
                "high_threshold",
                "suspicious",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect attacker query streams with CAP embedding-space coverage."
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
    parser.add_argument(
        "--group-mode",
        choices=["stream", "global"],
        default="stream",
        help=(
            "stream samples independent virtual users/query streams; global "
            "accumulates each split as one platform-wide query stream."
        ),
    )
    parser.add_argument("--normal-train-ratio", type=float, default=0.8)
    parser.add_argument("--max-normal", type=int, default=None)
    parser.add_argument("--max-attacker", type=int, default=None)
    parser.add_argument("--calibration-streams", type=int, default=100)
    parser.add_argument("--stream-length-batches", type=int, default=10)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--threshold", type=float, default=None, help="Override calibrated CAP cost threshold.")
    parser.add_argument(
        "--tail",
        choices=["high", "low", "two-sided"],
        default="high",
        help="Detection tail. high reproduces the original CAP high-coverage rule.",
    )
    parser.add_argument("--benign-eval-streams", type=int, default=20)
    parser.add_argument("--attacker-eval-streams", type=int, default=20)
    parser.add_argument(
        "--mixed-attacker-ratios",
        default="0.05,0.1,0.25,0.5",
        help="Comma-separated attacker ratios for mixed incoming streams. Use empty string to disable.",
    )
    parser.add_argument("--mixed-streams-per-ratio", type=int, default=20)
    parser.add_argument("--keep-last-batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("MMD_detection/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("CAP/outputs"))

    parser.add_argument("--num-buckets", type=int, default=2048)
    parser.add_argument("--baseline-lambda", type=float, default=0.0005)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--beta", type=float, default=0.19)
    parser.add_argument("--coverage-weight", type=float, default=0.05)
    parser.add_argument("--novelty-weight", type=float, default=0.35)
    parser.add_argument("--spread-weight", type=float, default=0.45)
    parser.add_argument("--spread-max", type=float, default=1.0)
    parser.add_argument("--max-cost", type=float, default=1e12)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    resolved_dataset = args.dataset
    if args.normal_path is None and args.attacker_path is None and args.normal_source == "dataset":
        normal_path, attacker_path, resolved_dataset = base.default_dataset_paths(
            args.data_root, args.dataset
        )
    else:
        if args.attacker_path is not None:
            attacker_path = args.attacker_path
        else:
            attacker_path, resolved_dataset = base.default_attacker_path(args.data_root, args.dataset)
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
    return normal_path, attacker_path, resolved_dataset


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
    base.np = _np

    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.calibration_streams < 1:
        raise ValueError("--calibration-streams must be at least 1")
    if args.stream_length_batches < 1:
        raise ValueError("--stream-length-batches must be at least 1")
    if not 50.0 < args.threshold_percentile < 100.0:
        raise ValueError("--threshold-percentile must be between 50 and 100")

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    normal_path, attacker_path, resolved_dataset = resolve_paths(args)
    mixed_ratios = parse_float_list(args.mixed_attacker_ratios)
    for ratio in mixed_ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError("--mixed-attacker-ratios values must be between 0 and 1")

    normal_queries = base.read_jsonl_queries(normal_path, args.text_field, args.max_normal)
    attacker_queries = base.read_jsonl_queries(attacker_path, args.text_field, args.max_attacker)
    rng.shuffle(normal_queries)
    rng.shuffle(attacker_queries)

    split_at = int(len(normal_queries) * args.normal_train_ratio)
    benign_calib_queries = normal_queries[:split_at]
    benign_test_queries = normal_queries[split_at:]
    if len(benign_calib_queries) == 0:
        raise ValueError("Benign calibration pool is empty")
    if len(benign_test_queries) == 0:
        raise ValueError("Held-out benign test stream is empty")
    if len(attacker_queries) == 0:
        raise ValueError("Attacker stream is empty")

    all_queries = benign_calib_queries + benign_test_queries + attacker_queries
    embeddings = base.embed_texts(
        all_queries,
        model_name=args.embedding_model,
        batch_size=args.encode_batch_size,
        device=args.device,
        prefix=args.query_prefix,
        normalize=not args.no_normalize_embeddings,
        cache_dir=args.cache_dir,
    )

    n_c = len(benign_calib_queries)
    n_h = len(benign_test_queries)
    z_calib = embeddings[:n_c]
    z_benign_test = embeddings[n_c : n_c + n_h]
    z_attacker = embeddings[n_c + n_h :]

    config = CAPConfig(
        num_buckets=args.num_buckets,
        baseline_lambda=args.baseline_lambda,
        alpha=args.alpha,
        beta=args.beta,
        coverage_weight=args.coverage_weight,
        novelty_weight=args.novelty_weight,
        spread_weight=args.spread_weight,
        spread_max=args.spread_max,
        max_cost=args.max_cost,
    )
    num_bits = int(math.ceil(math.log2(config.num_buckets)))
    random_planes = rng.normal(size=(embeddings.shape[1], num_bits)).astype(np.float32)

    calibration_costs = build_benign_calibration_costs(
        z_calib=z_calib,
        batch_size=args.batch_size,
        calibration_streams=args.calibration_streams,
        stream_length_batches=args.stream_length_batches,
        config=config,
        random_planes=random_planes,
        rng=rng,
    )
    high_threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(np.percentile(calibration_costs, args.threshold_percentile))
    )
    low_percentile = 100.0 - args.threshold_percentile
    low_threshold = (
        float(args.threshold)
        if args.threshold is not None and args.tail == "low"
        else float(np.percentile(calibration_costs, low_percentile))
    )

    results: list[CAPBatchScore] = []
    if args.group_mode == "global":
        results.extend(
            score_stream(
                split="global_heldout_benign",
                label=0,
                z_stream=z_benign_test,
                batch_size=args.batch_size,
                config=config,
                random_planes=random_planes,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                tail=args.tail,
                drop_last=not args.keep_last_batch,
            )
        )
        results.extend(
            score_stream(
                split="global_attacker",
                label=1,
                z_stream=z_attacker,
                batch_size=args.batch_size,
                drop_last=not args.keep_last_batch,
                config=config,
                random_planes=random_planes,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                tail=args.tail,
            )
        )
        if mixed_ratios:
            results.extend(
                score_global_mixed_stream(
                    z_benign_stream=z_benign_test,
                    z_attacker_stream=z_attacker,
                    batch_size=args.batch_size,
                    attacker_ratios=mixed_ratios,
                    config=config,
                    random_planes=random_planes,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    tail=args.tail,
                    rng=rng,
                    drop_last=not args.keep_last_batch,
                )
            )
    else:
        if args.benign_eval_streams > 0:
            results.extend(
                score_random_streams(
                    split="heldout_benign",
                    label=0,
                    z_stream=z_benign_test,
                    batch_size=args.batch_size,
                    num_streams=args.benign_eval_streams,
                    stream_length_batches=args.stream_length_batches,
                    config=config,
                    random_planes=random_planes,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    tail=args.tail,
                    rng=rng,
                )
            )
        else:
            results.extend(
                score_stream(
                    split="heldout_benign",
                    label=0,
                    z_stream=z_benign_test,
                    batch_size=args.batch_size,
                    drop_last=not args.keep_last_batch,
                    config=config,
                    random_planes=random_planes,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    tail=args.tail,
                )
            )
        if args.attacker_eval_streams > 0:
            results.extend(
                score_random_streams(
                    split="attacker",
                    label=1,
                    z_stream=z_attacker,
                    batch_size=args.batch_size,
                    num_streams=args.attacker_eval_streams,
                    stream_length_batches=args.stream_length_batches,
                    config=config,
                    random_planes=random_planes,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    tail=args.tail,
                    rng=rng,
                )
            )
        else:
            results.extend(
                score_stream(
                    split="attacker",
                    label=1,
                    z_stream=z_attacker,
                    batch_size=args.batch_size,
                    drop_last=not args.keep_last_batch,
                    config=config,
                    random_planes=random_planes,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    tail=args.tail,
                )
            )
        if mixed_ratios:
            results.extend(
                score_mixed_streams(
                    z_benign_stream=z_benign_test,
                    z_attacker_stream=z_attacker,
                    batch_size=args.batch_size,
                    attacker_ratios=mixed_ratios,
                    streams_per_ratio=args.mixed_streams_per_ratio,
                    stream_length_batches=args.stream_length_batches,
                    config=config,
                    random_planes=random_planes,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    tail=args.tail,
                    rng=rng,
                )
            )

    metrics = summarize(results)
    metrics_by_family = summarize_by_family(results)
    metadata = {
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
        "group_mode": args.group_mode,
        "normal_train_ratio": args.normal_train_ratio,
        "calibration_streams": args.calibration_streams,
        "stream_length_batches": args.stream_length_batches,
        "tail": args.tail,
        "threshold_percentile": args.threshold_percentile,
        "low_threshold_percentile": low_percentile,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "threshold": high_threshold,
        "normal_queries": len(normal_queries),
        "benign_calibration_size": len(benign_calib_queries),
        "heldout_benign_size": len(benign_test_queries),
        "attacker_queries": len(attacker_queries),
        "benign_eval_streams": args.benign_eval_streams,
        "attacker_eval_streams": args.attacker_eval_streams,
        "mixed_attacker_ratios": mixed_ratios,
        "mixed_streams_per_ratio": args.mixed_streams_per_ratio,
        "seed": args.seed,
        "cap_config": config.__dict__,
        "calibration_cost_mean": float(np.mean(calibration_costs)),
        "calibration_cost_std": float(np.std(calibration_costs, ddof=1)),
        "metrics": metrics,
        "metrics_by_family": metrics_by_family,
    }
    write_outputs(args.output_dir, results, metadata)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir / 'batch_scores.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
