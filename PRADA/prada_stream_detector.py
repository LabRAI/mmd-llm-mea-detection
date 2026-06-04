#!/usr/bin/env python3
"""Stream-level PRADA-style detection on query embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import prada_detector as base

np = None
torch = None


@dataclass
class StreamScore:
    split: str
    stream_id: int
    label: int
    size: int
    attacker_ratio: float
    attacker_count: int
    score: float
    min_w_statistic: float
    detection_step: int
    suspicious: bool
    distances_seen: int
    growing_set_size: int


def distance_to_growing_set(x, growing_set, metric: str) -> float:
    if metric == "cosine":
        x_norm = torch.nn.functional.normalize(x.unsqueeze(0), p=2, dim=1, eps=1e-12)
        g_norm = torch.nn.functional.normalize(growing_set, p=2, dim=1, eps=1e-12)
        distances = 1.0 - g_norm @ x_norm.squeeze(0)
        return float(torch.clamp(torch.min(distances), min=0.0).detach().cpu())
    if metric == "l2":
        return float(torch.min(torch.linalg.norm(growing_set - x, dim=1)).detach().cpu())
    raise ValueError(f"Unsupported distance metric: {metric}")


def stream_distance_matrix(stream_t, metric: str):
    if metric == "cosine":
        z = torch.nn.functional.normalize(stream_t, p=2, dim=1, eps=1e-12)
        return torch.clamp(1.0 - z @ z.T, min=0.0)
    if metric == "l2":
        return torch.cdist(stream_t, stream_t, p=2)
    raise ValueError(f"Unsupported distance metric: {metric}")


def shapiro_w(values) -> float:
    try:
        from scipy.stats import shapiro
    except ImportError as exc:
        raise SystemExit("Missing dependency: scipy.") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, _ = shapiro(values)
    return float(stat)


def prada_stream_trace(
    stream,
    metric: str,
    compute_device: str,
    delta: float | None,
    min_distances: int,
    max_shapiro_samples: int,
    shapiro_interval: int,
    precompute_distances: bool,
    stop_on_detection: bool,
    rng,
) -> tuple[float, int, int, int, list[float]]:
    growing_distances: list[float] = []
    distances: list[float] = []
    threshold = 0.0
    min_w = 1.0
    detection_step = -1
    w_trace: list[float] = []
    stream_t = torch.as_tensor(stream, dtype=torch.float32, device=compute_device)
    all_distances = stream_distance_matrix(stream_t, metric) if precompute_distances else None
    growing_indices = torch.empty(len(stream_t), dtype=torch.long, device=compute_device)
    growing = torch.empty_like(stream_t) if not precompute_distances else None
    growing_size = 0

    for step in range(1, len(stream_t) + 1):
        x = stream_t[step - 1]
        if growing_size == 0:
            if precompute_distances:
                growing_indices[growing_size] = step - 1
            else:
                growing[growing_size] = x
            growing_size += 1
            growing_distances.append(0.0)
            continue

        if precompute_distances:
            d_min = float(
                torch.min(all_distances[step - 1, growing_indices[:growing_size]]).detach().cpu()
            )
        else:
            d_min = distance_to_growing_set(x, growing[:growing_size], metric)
        distances.append(d_min)

        if d_min > threshold:
            if precompute_distances:
                growing_indices[growing_size] = step - 1
            else:
                growing[growing_size] = x
            growing_size += 1
            growing_distances.append(d_min)
            dg = np.asarray(growing_distances, dtype=np.float64)
            next_threshold = float(np.mean(dg) - np.std(dg))
            threshold = max(threshold, next_threshold)

        should_test = (
            len(distances) > min_distances
            and (len(distances) - min_distances) % shapiro_interval == 0
        )
        if should_test:
            d = np.asarray(distances, dtype=np.float64)
            d = base.trim_outliers_3sigma(d)
            d = base.subsample_for_shapiro(d, max_shapiro_samples, rng)
            if len(d) >= 3:
                w_statistic = shapiro_w(d)
                min_w = min(min_w, w_statistic)
                w_trace.append(w_statistic)
                if delta is not None and detection_step < 0 and w_statistic < delta:
                    detection_step = step
                    if stop_on_detection:
                        return min_w, detection_step, len(distances), growing_size, w_trace

    return min_w, detection_step, len(distances), growing_size, w_trace


def sample_stream(z, length: int, rng):
    replace = len(z) < length
    idx = rng.choice(len(z), size=length, replace=replace)
    return z[idx]


def iter_sequential_streams(z, stream_length: int, drop_last: bool):
    for start in range(0, len(z), stream_length):
        stream = z[start : start + stream_length]
        if len(stream) < stream_length and drop_last:
            continue
        if len(stream) >= 3:
            yield stream


def build_calibration_scores(
    z_b,
    stream_length: int,
    calibration_streams: int,
    metric: str,
    compute_device: str,
    min_distances: int,
    max_shapiro_samples: int,
    shapiro_interval: int,
    precompute_distances: bool,
    rng,
):
    scores = np.empty(calibration_streams, dtype=np.float64)
    for i in range(calibration_streams):
        stream = sample_stream(z_b, stream_length, rng)
        min_w, _, _, _, _ = prada_stream_trace(
            stream,
            metric,
            compute_device,
            None,
            min_distances,
            max_shapiro_samples,
            shapiro_interval,
            precompute_distances,
            False,
            rng,
        )
        scores[i] = 1.0 - min_w
    return scores


def evaluate_streams(
    split: str,
    label: int,
    streams,
    metric: str,
    compute_device: str,
    delta: float,
    min_distances: int,
    max_shapiro_samples: int,
    shapiro_interval: int,
    precompute_distances: bool,
    stop_on_detection: bool,
    rng,
    attacker_ratio: float | None = None,
    attacker_count: int | None = None,
) -> list[StreamScore]:
    results: list[StreamScore] = []
    for stream_id, stream in enumerate(streams):
        min_w, detection_step, distances_seen, growing_set_size, _ = prada_stream_trace(
            stream,
            metric,
            compute_device,
            delta,
            min_distances,
            max_shapiro_samples,
            shapiro_interval,
            precompute_distances,
            stop_on_detection,
            rng,
        )
        size = len(stream)
        ratio = float(label) if attacker_ratio is None else attacker_ratio
        count = size if label == 1 else 0
        if attacker_count is not None:
            count = attacker_count
        results.append(
            StreamScore(
                split=split,
                stream_id=stream_id,
                label=label,
                size=size,
                attacker_ratio=ratio,
                attacker_count=count,
                score=1.0 - min_w,
                min_w_statistic=min_w,
                detection_step=detection_step,
                suspicious=detection_step >= 0,
                distances_seen=distances_seen,
                growing_set_size=growing_set_size,
            )
        )
    return results


def evaluate_mixed_streams(
    z_benign,
    z_attacker,
    stream_length: int,
    attacker_ratios: list[float],
    streams_per_ratio: int,
    metric: str,
    compute_device: str,
    delta: float,
    min_distances: int,
    max_shapiro_samples: int,
    shapiro_interval: int,
    precompute_distances: bool,
    stop_on_detection: bool,
    rng,
) -> list[StreamScore]:
    results: list[StreamScore] = []
    for ratio in attacker_ratios:
        attacker_count = min(max(int(round(stream_length * ratio)), 1), stream_length - 1)
        benign_count = stream_length - attacker_count
        streams = []
        for _ in range(streams_per_ratio):
            benign = sample_stream(z_benign, benign_count, rng)
            attacker = sample_stream(z_attacker, attacker_count, rng)
            stream = np.concatenate([benign, attacker], axis=0)
            stream = stream[rng.permutation(len(stream))]
            streams.append(stream)
        results.extend(
            evaluate_streams(
                split=f"mixed_{ratio:g}",
                label=1,
                streams=streams,
                metric=metric,
                compute_device=compute_device,
                delta=delta,
                min_distances=min_distances,
                max_shapiro_samples=max_shapiro_samples,
                shapiro_interval=shapiro_interval,
                precompute_distances=precompute_distances,
                stop_on_detection=stop_on_detection,
                rng=rng,
                attacker_ratio=attacker_count / stream_length,
                attacker_count=attacker_count,
            )
        )
    return results


def summarize(results: list[StreamScore]) -> dict[str, float | int]:
    positives = [r for r in results if r.label == 1]
    negatives = [r for r in results if r.label == 0]
    tp = sum(r.suspicious for r in positives)
    fp = sum(r.suspicious for r in negatives)
    tn = len(negatives) - fp
    fn = len(positives) - tp
    total = len(results)
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


def summarize_by_split(results: list[StreamScore]) -> dict[str, dict[str, float | int]]:
    out = {}
    for split in sorted({r.split for r in results}):
        rs = [r for r in results if r.split == split]
        suspicious = sum(r.suspicious for r in rs)
        detected = [r.detection_step for r in rs if r.detection_step >= 0]
        out[split] = {
            "streams": len(rs),
            "suspicious": suspicious,
            "detection_rate": suspicious / len(rs) if rs else float("nan"),
            "mean_score": float(np.mean([r.score for r in rs])) if rs else float("nan"),
            "mean_min_w_statistic": float(np.mean([r.min_w_statistic for r in rs])) if rs else float("nan"),
            "mean_detection_step": float(np.mean(detected)) if detected else float("nan"),
            "mean_attacker_ratio": float(np.mean([r.attacker_ratio for r in rs])) if rs else float("nan"),
        }
    return out


def write_outputs(output_dir: Path, results: list[StreamScore], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stream_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "stream_id",
            "label",
            "size",
            "attacker_ratio",
            "attacker_count",
            "score",
            "min_w_statistic",
            "detection_step",
            "suspicious",
            "distances_seen",
            "growing_set_size",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-level PRADA-style detector over query embeddings."
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
    parser.add_argument(
        "--compute-device",
        default=None,
        help="Torch device for growing-set distance computation. Defaults to --device.",
    )
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--stream-length", type=int, default=6000)
    parser.add_argument("--calibration-streams", type=int, default=200)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--normal-train-ratio", type=float, default=0.8)
    parser.add_argument("--distance-metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--min-distances", type=int, default=100)
    parser.add_argument("--max-shapiro-samples", type=int, default=5000)
    parser.add_argument(
        "--shapiro-interval",
        type=int,
        default=1,
        help="Run Shapiro-Wilk every N new distances after --min-distances. 1 matches the original per-query check.",
    )
    parser.add_argument(
        "--no-precompute-distances",
        action="store_true",
        help="Disable per-stream GPU distance-matrix precomputation to reduce peak memory.",
    )
    parser.add_argument("--benign-eval-streams", type=int, default=50)
    parser.add_argument("--attacker-eval-streams", type=int, default=50)
    parser.add_argument("--mixed-attacker-ratios", default="")
    parser.add_argument("--mixed-streams-per-ratio", type=int, default=50)
    parser.add_argument(
        "--no-stop-on-detection",
        action="store_true",
        help="Continue scoring a stream after the first PRADA alarm. Default stops at first alarm.",
    )
    parser.add_argument("--sequential-eval", action="store_true")
    parser.add_argument("--keep-last-stream", action="store_true")
    parser.add_argument("--max-normal", type=int, default=None)
    parser.add_argument("--max-attacker", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("PRADA/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("PRADA/stream_outputs"))
    return parser.parse_args()


def main() -> None:
    global np
    import numpy as _np

    np = _np
    base.np = _np
    args = parse_args()
    try:
        global torch
        import torch as _torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch.") from exc
    torch = _torch
    if args.stream_length <= args.min_distances + 2:
        raise ValueError("--stream-length must be larger than --min-distances + 2")
    if args.shapiro_interval < 1:
        raise ValueError("--shapiro-interval must be at least 1")

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    compute_device = args.compute_device or args.device

    resolved_dataset = args.dataset
    if args.normal_path is None and args.attacker_path is None and args.normal_source == "dataset":
        normal_path, attacker_path, resolved_dataset = base.default_dataset_paths(args.data_root, args.dataset)
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
            raise FileNotFoundError(f"Missing normal queries: {normal_path}")

    mixed_ratios = [float(x.strip()) for x in args.mixed_attacker_ratios.split(",") if x.strip()]

    normal_queries = base.read_jsonl_queries(normal_path, args.text_field, args.max_normal)
    attacker_queries = base.read_jsonl_queries(attacker_path, args.text_field, args.max_attacker)
    rng.shuffle(normal_queries)
    rng.shuffle(attacker_queries)

    split_at = int(len(normal_queries) * args.normal_train_ratio)
    benign_pool_queries = normal_queries[:split_at]
    benign_test_queries = normal_queries[split_at:]
    if len(benign_pool_queries) < 3:
        raise ValueError("Benign calibration pool has fewer than 3 queries")
    if len(benign_test_queries) < 3:
        raise ValueError("Held-out benign stream has fewer than 3 queries")

    all_queries = benign_pool_queries + benign_test_queries + attacker_queries
    embeddings = base.embed_texts(
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

    calibration_scores = build_calibration_scores(
        z_b,
        args.stream_length,
        args.calibration_streams,
        args.distance_metric,
        compute_device,
        args.min_distances,
        args.max_shapiro_samples,
        args.shapiro_interval,
        not args.no_precompute_distances,
        rng,
    )
    threshold_score = float(np.percentile(calibration_scores, args.threshold_percentile))
    delta = 1.0 - threshold_score

    if args.sequential_eval:
        benign_streams = list(iter_sequential_streams(z_benign_test, args.stream_length, not args.keep_last_stream))
        attacker_streams = list(iter_sequential_streams(z_attacker, args.stream_length, not args.keep_last_stream))
    else:
        benign_streams = [sample_stream(z_benign_test, args.stream_length, rng) for _ in range(args.benign_eval_streams)]
        attacker_streams = [sample_stream(z_attacker, args.stream_length, rng) for _ in range(args.attacker_eval_streams)]

    results: list[StreamScore] = []
    results.extend(
        evaluate_streams(
            "heldout_benign",
            0,
            benign_streams,
            args.distance_metric,
            compute_device,
            delta,
            args.min_distances,
            args.max_shapiro_samples,
            args.shapiro_interval,
            not args.no_precompute_distances,
            not args.no_stop_on_detection,
            rng,
        )
    )
    results.extend(
        evaluate_streams(
            "attacker",
            1,
            attacker_streams,
            args.distance_metric,
            compute_device,
            delta,
            args.min_distances,
            args.max_shapiro_samples,
            args.shapiro_interval,
            not args.no_precompute_distances,
            not args.no_stop_on_detection,
            rng,
        )
    )
    if mixed_ratios:
        results.extend(
            evaluate_mixed_streams(
                z_benign_test,
                z_attacker,
                args.stream_length,
                mixed_ratios,
                args.mixed_streams_per_ratio,
                args.distance_metric,
                compute_device,
                delta,
                args.min_distances,
                args.max_shapiro_samples,
                args.shapiro_interval,
                not args.no_precompute_distances,
                not args.no_stop_on_detection,
                rng,
            )
        )

    metadata = {
        "method": "stream_prada_single_group_shapiro",
        "note": (
            "Original PRADA is client-level and class-conditioned. This reproduces the "
            "stateful stream/growing-set logic, using a single group because these LLM "
            "query datasets do not provide target-model predicted classes."
        ),
        "dataset": resolved_dataset,
        "normal_source": args.normal_source,
        "normal_path": str(normal_path),
        "attacker_path": str(attacker_path),
        "embedding_model": args.embedding_model,
        "device": args.device,
        "compute_device": compute_device,
        "normalize_embeddings": not args.no_normalize_embeddings,
        "distance_metric": args.distance_metric,
        "stream_length": args.stream_length,
        "min_distances": args.min_distances,
        "shapiro_interval": args.shapiro_interval,
        "precompute_distances": not args.no_precompute_distances,
        "stop_on_detection": not args.no_stop_on_detection,
        "calibration_streams": args.calibration_streams,
        "threshold_percentile": args.threshold_percentile,
        "threshold_score": threshold_score,
        "delta_w_threshold": delta,
        "normal_queries": len(normal_queries),
        "benign_pool_size": len(benign_pool_queries),
        "heldout_benign_size": len(benign_test_queries),
        "attacker_queries": len(attacker_queries),
        "mixed_attacker_ratios": mixed_ratios,
        "seed": args.seed,
        "metrics": summarize(results),
        "metrics_by_split": summarize_by_split(results),
    }
    write_outputs(args.output_dir, results, metadata)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir / 'stream_scores.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
