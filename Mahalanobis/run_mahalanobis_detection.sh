#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET="${DATASET:-model_leeching}"
DATA_ROOT="${DATA_ROOT:-data}"
NORMAL_SOURCE="${NORMAL_SOURCE:-dataset}"
#NORMAL_SOURCE="${NORMAL_SOURCE:-global}"
GLOBAL_NORMAL_PATH="${GLOBAL_NORMAL_PATH:-data/global_normal/queries.jsonl}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
#EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-roberta-base-v1}"
DEVICE="${DEVICE:-}"

BATCH_SIZE="${BATCH_SIZE:-1500}"
NULL_SAMPLES="${NULL_SAMPLES:-1000}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.7}"
NORMAL_CALIBRATION_RATIO="${NORMAL_CALIBRATION_RATIO:-0.1}"
MAHA_RIDGE="${MAHA_RIDGE:-1e-6}"
MAHA_AGGREGATION="${MAHA_AGGREGATION:-mean}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-0.05,0.1,0.25,0.5}"
MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO:-50}"
BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES:-50}"
ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES:-50}"
SEED="${SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-Mahalanobis/outputs/${DATASET}_bge_bs${BATCH_SIZE}}"
CACHE_DIR="${CACHE_DIR:-Mahalanobis/cache}"

cmd=(
  python3 Mahalanobis/mahalanobis_detector.py
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --normal-source "${NORMAL_SOURCE}"
  --global-normal-path "${GLOBAL_NORMAL_PATH}"
  --embedding-model "${EMBEDDING_MODEL}"
  --batch-size "${BATCH_SIZE}"
  --null-samples "${NULL_SAMPLES}"
  --threshold-percentile "${THRESHOLD_PERCENTILE}"
  --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
  --normal-calibration-ratio "${NORMAL_CALIBRATION_RATIO}"
  --maha-ridge "${MAHA_RIDGE}"
  --maha-aggregation "${MAHA_AGGREGATION}"
  --mixed-attacker-ratios "${MIXED_ATTACKER_RATIOS}"
  --mixed-batches-per-ratio "${MIXED_BATCHES_PER_RATIO}"
  --benign-eval-batches "${BENIGN_EVAL_BATCHES}"
  --attacker-eval-batches "${ATTACKER_EVAL_BATCHES}"
  --seed "${SEED}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

echo "Running marginal Mahalanobis detection..."
echo "Dataset: ${DATASET}"
echo "Normal source: ${NORMAL_SOURCE}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Aggregation: ${MAHA_AGGREGATION}"
echo "Ridge: ${MAHA_RIDGE}"
echo "Normal split: fit=${NORMAL_TRAIN_RATIO}, calibration=${NORMAL_CALIBRATION_RATIO}, heldout=remaining"
echo "Output dir: ${OUTPUT_DIR}"
echo

"${cmd[@]}"
