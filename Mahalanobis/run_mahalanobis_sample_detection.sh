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

THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.7}"
NORMAL_CALIBRATION_RATIO="${NORMAL_CALIBRATION_RATIO:-0.1}"
MAHA_RIDGE="${MAHA_RIDGE:-1e-6}"
SEED="${SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-Mahalanobis/sample_outputs/${DATASET}_bge_sample}"
CACHE_DIR="${CACHE_DIR:-Mahalanobis/cache}"

cmd=(
  python3 Mahalanobis/mahalanobis_sample_detector.py
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --normal-source "${NORMAL_SOURCE}"
  --global-normal-path "${GLOBAL_NORMAL_PATH}"
  --embedding-model "${EMBEDDING_MODEL}"
  --threshold-percentile "${THRESHOLD_PERCENTILE}"
  --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
  --normal-calibration-ratio "${NORMAL_CALIBRATION_RATIO}"
  --maha-ridge "${MAHA_RIDGE}"
  --seed "${SEED}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

echo "Running sample-level marginal Mahalanobis detection..."
echo "Dataset: ${DATASET}"
echo "Normal source: ${NORMAL_SOURCE}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Ridge: ${MAHA_RIDGE}"
echo "Normal split: fit=${NORMAL_TRAIN_RATIO}, calibration=${NORMAL_CALIBRATION_RATIO}, heldout=remaining"
echo "Output dir: ${OUTPUT_DIR}"
echo

"${cmd[@]}"
