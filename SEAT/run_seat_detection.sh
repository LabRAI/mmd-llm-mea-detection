#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET="${DATASET:-model_leeching}"
DATA_ROOT="${DATA_ROOT:-data}"
NORMAL_SOURCE="${NORMAL_SOURCE:-dataset}"
GLOBAL_NORMAL_PATH="${GLOBAL_NORMAL_PATH:-data/global_normal/queries.jsonl}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
DEVICE="${DEVICE:-cuda}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-${DEVICE}}"

BATCH_SIZE="${BATCH_SIZE:-1500}"
SCORE_MODE="${SCORE_MODE:-ratio}"
DETECTION_TAIL="${DETECTION_TAIL:-two-sided}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-}"
SIMILARITY_THRESHOLD_PERCENTILE="${SIMILARITY_THRESHOLD_PERCENTILE:-99}"
PAIR_SAMPLE_SIZE="${PAIR_SAMPLE_SIZE:-200000}"
NULL_SAMPLES="${NULL_SAMPLES:-1000}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-0.05,0.1,0.25,0.5}"
MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO:-50}"
BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES:-50}"
ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES:-50}"
SEED="${SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-SEAT/outputs/${DATASET}_bge_bs${BATCH_SIZE}}"
CACHE_DIR="${CACHE_DIR:-SEAT/cache}"

cmd=(
  python3 SEAT/seat_detector.py
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --normal-source "${NORMAL_SOURCE}"
  --global-normal-path "${GLOBAL_NORMAL_PATH}"
  --embedding-model "${EMBEDDING_MODEL}"
  --batch-size "${BATCH_SIZE}"
  --score-mode "${SCORE_MODE}"
  --detection-tail "${DETECTION_TAIL}"
  --similarity-threshold-percentile "${SIMILARITY_THRESHOLD_PERCENTILE}"
  --pair-sample-size "${PAIR_SAMPLE_SIZE}"
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
)

if [[ -n "${SIMILARITY_THRESHOLD}" ]]; then
  cmd+=(--similarity-threshold "${SIMILARITY_THRESHOLD}")
fi

if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

if [[ -n "${COMPUTE_DEVICE}" ]]; then
  cmd+=(--compute-device "${COMPUTE_DEVICE}")
fi

echo "Running SEAT-style detection..."
echo "Dataset: ${DATASET}"
echo "Normal source: ${NORMAL_SOURCE}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Device: ${DEVICE}"
echo "Compute device: ${COMPUTE_DEVICE}"
echo "Score mode: ${SCORE_MODE}"
echo "Detection tail: ${DETECTION_TAIL}"
echo "Output dir: ${OUTPUT_DIR}"
echo

"${cmd[@]}"
