#!/usr/bin/env python3
"""Original-style CAP detector over cumulative query streams.

This script keeps CAP close to the paper setting: each sampled stream represents
one user/account over time, the detector accumulates coverage across multiple
batches, and a stream is detected if its cumulative CAP cost crosses a calibrated
high-tail benign threshold at any point.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAP import cap_detector as cap

np = None

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = object


@dataclass
class CAPStreamScore:
    split: str
    stream_id: int
    label: int
    attacker_ratio: float
    attacker_count_per_batch: int
    batch_size: int
    stream_length_batches: int
    total_queries: int
    detected: bool
    alarm_batch: int
    alarm_queries: int
    max_cost: float
    final_cost: float
    final_bucket_coverage: float
    final_new_bucket_rate: float
    final_spread: float
    threshold: float


def run_stream(
    batches: list[NDArray],
    label: int,
    split: str,
    stream_id: int,
    attacker_ratio: float,
    attacker_count_per_batch: int,
    config: cap.CAPConfig,
    random_planes: NDArray,
    threshold: float,
) -> CAPStreamScore:
    detector = cap.make_detector(batches[0].shape[1], config, random_planes)
    max_cost = float("-inf")
    final_cost = 0.0
    final_coverage = 0.0
    final_new_bucket_rate = 0.0
    final_spread = 0.0
    total_queries = 0
    alarm_batch = -1
    alarm_queries = -1

    for batch_id, batch in enumerate(batches, start=1):
        coverage, new_bucket_rate, spread, cost, total_queries = detector.update(batch)
        max_cost = max(max_cost, cost)
        final_cost = cost
        final_coverage = coverage
        final_new_bucket_rate = new_bucket_rate
        final_spread = spread
        if alarm_batch < 0 and cost > threshold:
            alarm_batch = batch_id
            alarm_queries = total_queries

    return CAPStreamScore(
        split=split,
        stream_id=stream_id,
        label=label,
        attacker_ratio=attacker_ratio,
        attacker_count_per_batch=attacker_count_per_batch,
        batch_size=len(batches[0]),
        stream_length_batches=len(batches),
        total_queries=total_queries,
        detected=alarm_batch > 0,
        alarm_batch=alarm_batch,
        alarm_queries=alarm_queries,
        max_cost=max_cost,
        final_cost=final_cost,
        final_bucket_coverage=final_coverage,
        final_new_bucket_rate=final_new_bucket_rate,
        final_spread=final_spread,
        threshold=threshold,
    )


def sample_stream_batches(
    z_stream: NDArray,
    batch_size: int,
    stream_length_batches: int,
    rng: object,
) -> list[NDArray]:
    return [cap.base.sample_batch(z_stream, batch_size, rng) for _ in range(stream_length_batches)]


def sample_mixed_stream_batches(
    z_benign_stream: NDArray,
    z_attacker_stream: NDArray,
    batch_size: int,
    attacker_ratio: float,
    stream_length_batches: int,
    rng: object,
) -> tuple[list[NDArray], int, float]:
    attacker_count = int(round(batch_size * attacker_ratio))
    attacker_count = min(max(attacker_count, 1), batch_size - 1)
    benign_count = batch_size - attacker_count
    actual_ratio = attacker_count / batch_size
    batches = []
    for _ in range(stream_length_batches):
        benign_part = cap.base.sample_batch(z_benign_stream, benign_count, rng)
        attacker_part = cap.base.sample_batch(z_attacker_stream, attacker_count, rng)
        batch = np.concatenate([benign_part, attacker_part], axis=0)
        batches.append(batch[rng.permutation(len(batch))])
    return batches, attacker_count, actual_ratio


def build_calibration_max_costs(
    z_calib: NDArray,
    batch_size: int,
    calibration_streams: int,
    stream_length_batches: int,
    config: cap.CAPConfig,
    random_planes: NDArray,
    rng: object,
) -> NDArray:
    scores = np.empty(calibration_streams, dtype=np.float64)
    for stream_id in range(calibration_streams):
        batches = sample_stream_batches(z_calib, batch_size, stream_length_batches, rng)
        result = run_stream(
            batches=batches,
            label=0,
            split="calibration",
            stream_id=stream_id,
            attacker_ratio=0.0,
            attacker_count_per_batch=0,
            config=config,
            random_planes=random_planes,
            threshold=float("inf"),
        )
        scores[stream_id] = result.max_cost
    return scores


def evaluate_random_streams(
    split: str,
    label: int,
    z_stream: NDArray,
    batch_size: int,
    num_streams: int,
    stream_length_batches: int,
    config: cap.CAPConfig,
    random_planes: NDArray,
    threshold: float,
    rng: object,
) -> list[CAPStreamScore]:
    results = []
    for stream_id in range(num_streams):
        batches = sample_stream_batches(z_stream, batch_size, stream_length_batches, rng)
        results.append(
            run_stream(
                batches=batches,
                label=label,
                split=split,
                stream_id=stream_id,
                attacker_ratio=float(label),
                attacker_count_per_batch=batch_size if label == 1 else 0,
                config=config,
                random_planes=random_planes,
                threshold=threshold,
            )
        )
    return results


def evaluate_mixed_streams(
    z_benign_stream: NDArray,
    z_attacker_stream: NDArray,
    batch_size: int,
    attacker_ratios: list[float],
    streams_per_ratio: int,
    stream_length_batches: int,
    config: cap.CAPConfig,
    random_planes: NDArray,
    threshold: float,
    rng: object,
) -> list[CAPStreamScore]:
    results = []
    for ratio in attacker_ratios:
        for stream_id in range(streams_per_ratio):
            batches, attacker_count, actual_ratio = sample_mixed_stream_batches(
                z_benign_stream=z_benign_stream,
                z_attacker_stream=z_attacker_stream,
                batch_size=batch_size,
                attacker_ratio=ratio,
                stream_length_batches=stream_length_batches,
                rng=rng,
            )
            results.append(
                run_stream(
                    batches=batches,
                    label=1,
                    split=f"mixed_{ratio:g}",
                    stream_id=stream_id,
                    attacker_ratio=actual_ratio,
                    attacker_count_per_batch=attacker_count,
                    config=config,
                    random_planes=random_planes,
                    threshold=threshold,
                )
            )
    return results


def summarize(results: list[CAPStreamScore]) -> dict[str, float | int]:
    total = len(results)
    positives = [r for r in results if r.label == 1]
    negatives = [r for r in results if r.label == 0]
    tp = sum(r.detected for r in positives)
    fp = sum(r.detected for r in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
    return {
        "streams": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / len(positives) if positives else float("nan"),
        "fpr": fp / len(negatives) if negatives else float("nan"),
        "accuracy": (tp + tn) / total if total else float("nan"),
    }


def summarize_by_split(results: list[CAPStreamScore]) -> dict[str, dict[str, float | int]]:
    summaries = {}
    for split in sorted({r.split for r in results}):
        split_results = [r for r in results if r.split == split]
        detected = sum(r.detected for r in split_results)
        alarm_queries = [r.alarm_queries for r in split_results if r.alarm_queries > 0]
        summaries[split] = {
            "streams": len(split_results),
            "detected": detected,
            "detection_rate": detected / len(split_results) if split_results else float("nan"),
            "mean_max_cost": (
                float(np.mean([r.max_cost for r in split_results])) if split_results else float("nan")
            ),
            "mean_final_cost": (
                float(np.mean([r.final_cost for r in split_results])) if split_results else float("nan")
            ),
            "mean_final_coverage": (
                float(np.mean([r.final_bucket_coverage for r in split_results]))
                if split_results
                else float("nan")
            ),
            "mean_alarm_queries": float(np.mean(alarm_queries)) if alarm_queries else float("nan"),
            "mean_attacker_ratio": (
                float(np.mean([r.attacker_ratio for r in split_results]))
                if split_results
                else float("nan")
            ),
        }
    return summaries


def write_outputs(output_dir: Path, results: list[CAPStreamScore], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stream_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CAPStreamScore.__dataclass_fields__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original-style CAP detector over cumulative virtual user streams."
    )
    parser.add_argument("--dataset", default="model_leeching")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--normal-source", choices=["dataset", "global"], default="dataset")
    parser.add_argument("--global-normal-path", type=Path, default=Path("data/global_normal/queries.jsonl"))
    parser.add_argument("--normal-path", type=Path, default=None)
    parser.add_argument("--attacker-path", type=Path, default=None)
    parser.add_argument("--text-field", default="query")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--stream-length-batches", type=int, default=10)
    parser.add_argument("--normal-train-ratio", type=float, default=0.8)
    parser.add_argument("--calibration-streams", type=int, default=1000)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--benign-eval-streams", type=int, default=50)
    parser.add_argument("--attacker-eval-streams", type=int, default=50)
    parser.add_argument("--mixed-attacker-ratios", default="0.05,0.1,0.25,0.5")
    parser.add_argument("--mixed-streams-per-ratio", type=int, default=50)
    parser.add_argument("--max-normal", type=int, default=None)
    parser.add_argument("--max-attacker", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("MMD_detection/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("CAP/outputs_original"))

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


def main() -> None:
    global np
    args = parse_args()
    try:
        import numpy as _np
    except ImportError as exc:
        raise SystemExit("Missing dependency: numpy.") from exc
    np = _np
    cap.np = _np
    cap.base.np = _np

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.stream_length_batches < 1:
        raise ValueError("--stream-length-batches must be at least 1")
    if args.calibration_streams < 1:
        raise ValueError("--calibration-streams must be at least 1")
    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    normal_path, attacker_path, resolved_dataset = cap.resolve_paths(args)
    mixed_ratios = cap.parse_float_list(args.mixed_attacker_ratios)
    for ratio in mixed_ratios:
        if not 0.0 < ratio < 1.0:
            raise ValueError("--mixed-attacker-ratios values must be between 0 and 1")

    normal_queries = cap.base.read_jsonl_queries(normal_path, args.text_field, args.max_normal)
    attacker_queries = cap.base.read_jsonl_queries(attacker_path, args.text_field, args.max_attacker)
    rng.shuffle(normal_queries)
    rng.shuffle(attacker_queries)

    split_at = int(len(normal_queries) * args.normal_train_ratio)
    benign_calib_queries = normal_queries[:split_at]
    benign_test_queries = normal_queries[split_at:]
    if not benign_calib_queries:
        raise ValueError("Benign calibration pool is empty")
    if not benign_test_queries:
        raise ValueError("Held-out benign test stream is empty")
    if not attacker_queries:
        raise ValueError("Attacker stream is empty")

    all_queries = benign_calib_queries + benign_test_queries + attacker_queries
    embeddings = cap.base.embed_texts(
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

    config = cap.CAPConfig(
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
    num_bits = int(np.ceil(np.log2(config.num_buckets)))
    random_planes = rng.normal(size=(embeddings.shape[1], num_bits)).astype(np.float32)

    calibration_max_costs = build_calibration_max_costs(
        z_calib=z_calib,
        batch_size=args.batch_size,
        calibration_streams=args.calibration_streams,
        stream_length_batches=args.stream_length_batches,
        config=config,
        random_planes=random_planes,
        rng=rng,
    )
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(np.percentile(calibration_max_costs, args.threshold_percentile))
    )

    results: list[CAPStreamScore] = []
    results.extend(
        evaluate_random_streams(
            split="heldout_benign",
            label=0,
            z_stream=z_benign_test,
            batch_size=args.batch_size,
            num_streams=args.benign_eval_streams,
            stream_length_batches=args.stream_length_batches,
            config=config,
            random_planes=random_planes,
            threshold=threshold,
            rng=rng,
        )
    )
    results.extend(
        evaluate_random_streams(
            split="attacker",
            label=1,
            z_stream=z_attacker,
            batch_size=args.batch_size,
            num_streams=args.attacker_eval_streams,
            stream_length_batches=args.stream_length_batches,
            config=config,
            random_planes=random_planes,
            threshold=threshold,
            rng=rng,
        )
    )
    if mixed_ratios:
        results.extend(
            evaluate_mixed_streams(
                z_benign_stream=z_benign_test,
                z_attacker_stream=z_attacker,
                batch_size=args.batch_size,
                attacker_ratios=mixed_ratios,
                streams_per_ratio=args.mixed_streams_per_ratio,
                stream_length_batches=args.stream_length_batches,
                config=config,
                random_planes=random_planes,
                threshold=threshold,
                rng=rng,
            )
        )

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
        "device": args.device,
        "detector_level": "original_stream",
        "tail": "high",
        "batch_size": args.batch_size,
        "stream_length_batches": args.stream_length_batches,
        "normal_train_ratio": args.normal_train_ratio,
        "calibration_streams": args.calibration_streams,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
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
        "calibration_max_cost_mean": float(np.mean(calibration_max_costs)),
        "calibration_max_cost_std": float(np.std(calibration_max_costs, ddof=1)),
        "metrics": summarize(results),
        "metrics_by_split": summarize_by_split(results),
    }
    write_outputs(args.output_dir, results, metadata)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir / 'stream_scores.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
