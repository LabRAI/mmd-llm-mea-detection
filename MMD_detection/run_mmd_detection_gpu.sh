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
DEVICE="${DEVICE:-cuda}"
MMD_DEVICE="${MMD_DEVICE:-cuda}"
MMD_DTYPE="${MMD_DTYPE:-float32}"

BATCH_SIZE="${BATCH_SIZE:-1500}"
REFERENCE_REPEATS="${REFERENCE_REPEATS:-20}"
NULL_SAMPLES="${NULL_SAMPLES:-1000}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-0.05,0.1,0.25,0.5}"
MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO:-50}"
#BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES:-0}"
BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES:-50}"
#ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES:-0}"
ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES:-50}"
SEED="${SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-MMD_detection/outputs_gpu/${DATASET}_bge_bs${BATCH_SIZE}}"
#OUTPUT_DIR="${OUTPUT_DIR:-MMD_detection/outputs_2/${DATASET}_bge_bs${BATCH_SIZE}}"  正式的实验结果
CACHE_DIR="${CACHE_DIR:-MMD_detection/cache}"

cmd=(
  python3 MMD_detection/mmd_detector_gpu.py
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --normal-source "${NORMAL_SOURCE}"
  --global-normal-path "${GLOBAL_NORMAL_PATH}"
  --embedding-model "${EMBEDDING_MODEL}"
  --mmd-device "${MMD_DEVICE}"
  --mmd-dtype "${MMD_DTYPE}"
  --batch-size "${BATCH_SIZE}"
  --reference-repeats "${REFERENCE_REPEATS}"
  --null-samples "${NULL_SAMPLES}"
  --threshold-percentile "${THRESHOLD_PERCENTILE}"
  --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
  --mixed-attacker-ratios "${MIXED_ATTACKER_RATIOS}"
  --mixed-batches-per-ratio "${MIXED_BATCHES_PER_RATIO}"
  --benign-eval-batches "${BENIGN_EVAL_BATCHES}"
  --attacker-eval-batches "${ATTACKER_EVAL_BATCHES}"
  --seed "${SEED}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --multi-kernel
)

if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

echo "Running GPU MMD detection..."
echo "Dataset: ${DATASET}"
echo "Normal source: ${NORMAL_SOURCE}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Embedding device: ${DEVICE}"
echo "MMD device: ${MMD_DEVICE}"
echo "Output dir: ${OUTPUT_DIR}"
echo

"${cmd[@]}"
