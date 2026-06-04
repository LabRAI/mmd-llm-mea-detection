#!/usr/bin/env python3
"""Sample-level OOD detection with marginal Mahalanobis distance.

This is the original granularity used by marginal Mahalanobis OOD scoring:
each query receives an OOD score independently. The batch-level detector in
mahalanobis_detector.py aggregates these scores over batches; this script keeps
the single-sample decision surface visible for direct comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import mahalanobis_detector as md

np = None


def empirical_p_values(scores, null_scores):
    return (np.sum(null_scores[:, None] >= scores[None, :], axis=0) + 1.0) / (len(null_scores) + 1.0)


def binary_metrics(labels, suspicious) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int32)
    suspicious = np.asarray(suspicious, dtype=bool)
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(suspicious & positives))
    fp = int(np.sum(suspicious & negatives))
    fn = int(np.sum((~suspicious) & positives))
    tn = int(np.sum((~suspicious) & negatives))
    total = len(labels)
    return {
        "samples": int(total),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / int(np.sum(positives)) if np.any(positives) else float("nan"),
        "fpr": fp / int(np.sum(negatives)) if np.any(negatives) else float("nan"),
        "accuracy": (tp + tn) / total if total else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
    }


def auroc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    n_pos = len(positives)
    n_neg = len(negatives)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def aupr_ood(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def summarize_by_split(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    summaries = {}
    for split in sorted({r["split"] for r in rows}):
        split_rows = [r for r in rows if r["split"] == split]
        scores = np.asarray([r["score"] for r in split_rows], dtype=np.float64)
        suspicious = np.asarray([r["suspicious"] for r in split_rows], dtype=bool)
        summaries[split] = {
            "samples": len(split_rows),
            "suspicious": int(np.sum(suspicious)),
            "detection_rate": float(np.mean(suspicious)) if len(suspicious) else float("nan"),
            "mean_score": float(np.mean(scores)) if len(scores) else float("nan"),
            "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        }
    return summaries


def write_outputs(output_dir: Path, rows: list[dict], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sample_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sample_id", "label", "score", "p_value", "suspicious", "query"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect attacker queries using sample-level marginal Mahalanobis distance."
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--normal-train-ratio", type=float, default=0.7)
    parser.add_argument("--normal-calibration-ratio", type=float, default=0.1)
    parser.add_argument("--maha-ridge", type=float, default=1e-6)
    parser.add_argument("--max-normal", type=int, default=None)
    parser.add_argument("--max-attacker", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("Mahalanobis/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("Mahalanobis/sample_outputs"))
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
    md.np = np

    if not 0.0 < args.normal_train_ratio < 1.0:
        raise ValueError("--normal-train-ratio must be between 0 and 1")
    if not 0.0 < args.normal_calibration_ratio < 1.0:
        raise ValueError("--normal-calibration-ratio must be between 0 and 1")
    if args.normal_train_ratio + args.normal_calibration_ratio >= 1.0:
        raise ValueError("--normal-train-ratio + --normal-calibration-ratio must be less than 1")

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    resolved_dataset = args.dataset
    if args.normal_path is None and args.attacker_path is None and args.normal_source == "dataset":
        normal_path, attacker_path, resolved_dataset = md.default_dataset_paths(args.data_root, args.dataset)
    else:
        if args.attacker_path is not None:
            attacker_path = args.attacker_path
        else:
            attacker_path, resolved_dataset = md.default_attacker_path(args.data_root, args.dataset)
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

    normal_queries = md.read_jsonl_queries(normal_path, args.text_field, args.max_normal)
    attacker_queries = md.read_jsonl_queries(attacker_path, args.text_field, args.max_attacker)
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
    embeddings = md.embed_texts(
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
    z_calibration = embeddings[n_b : n_b + n_c]
    z_heldout = embeddings[n_b + n_c : n_b + n_c + n_h]
    z_attacker = embeddings[n_b + n_c + n_h :]

    model = md.fit_marginal_mahalanobis(z_b, args.maha_ridge)
    null_scores = md.mahalanobis_distances(model, z_calibration)
    threshold = float(np.percentile(null_scores, args.threshold_percentile))

    heldout_scores = md.mahalanobis_distances(model, z_heldout)
    attacker_scores = md.mahalanobis_distances(model, z_attacker)
    heldout_p = empirical_p_values(heldout_scores, null_scores)
    attacker_p = empirical_p_values(attacker_scores, null_scores)

    rows = []
    for i, (query, score, p_value) in enumerate(zip(benign_test_queries, heldout_scores, heldout_p)):
        rows.append(
            {
                "split": "heldout_benign",
                "sample_id": i,
                "label": 0,
                "score": float(score),
                "p_value": float(p_value),
                "suspicious": bool(score > threshold),
                "query": query,
            }
        )
    for i, (query, score, p_value) in enumerate(zip(attacker_queries, attacker_scores, attacker_p)):
        rows.append(
            {
                "split": "attacker",
                "sample_id": i,
                "label": 1,
                "score": float(score),
                "p_value": float(p_value),
                "suspicious": bool(score > threshold),
                "query": query,
            }
        )

    labels = np.asarray([r["label"] for r in rows], dtype=np.int32)
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    suspicious = np.asarray([r["suspicious"] for r in rows], dtype=bool)
    metrics = binary_metrics(labels, suspicious)
    metrics["auroc"] = float(auroc(labels, scores))
    metrics["aupr_ood"] = float(aupr_ood(labels, scores))

    metadata = {
        "detector": "sample_level_marginal_mahalanobis",
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
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "normal_train_ratio": args.normal_train_ratio,
        "normal_calibration_ratio": args.normal_calibration_ratio,
        "maha_ridge": args.maha_ridge,
        "normal_queries": len(normal_queries),
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
        "seed": args.seed,
        "metrics": metrics,
        "metrics_by_split": summarize_by_split(rows),
    }
    write_outputs(args.output_dir, rows, metadata)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output_dir / 'sample_scores.csv'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
