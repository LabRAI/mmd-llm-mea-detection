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

BATCH_SIZE="${BATCH_SIZE:-100}"
STREAM_LENGTH_BATCHES="${STREAM_LENGTH_BATCHES:-10}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
CALIBRATION_STREAMS="${CALIBRATION_STREAMS:-1000}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
BENIGN_EVAL_STREAMS="${BENIGN_EVAL_STREAMS:-50}"
ATTACKER_EVAL_STREAMS="${ATTACKER_EVAL_STREAMS:-50}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-0.05,0.1,0.25,0.5}"
MIXED_STREAMS_PER_RATIO="${MIXED_STREAMS_PER_RATIO:-50}"
SEED="${SEED:-42}"

NUM_BUCKETS="${NUM_BUCKETS:-2048}"
BASELINE_LAMBDA="${BASELINE_LAMBDA:-0.0005}"
ALPHA="${ALPHA:-8.0}"
BETA="${BETA:-0.19}"
COVERAGE_WEIGHT="${COVERAGE_WEIGHT:-0.05}"
NOVELTY_WEIGHT="${NOVELTY_WEIGHT:-0.35}"
SPREAD_WEIGHT="${SPREAD_WEIGHT:-0.45}"
SPREAD_MAX="${SPREAD_MAX:-1.0}"

OUTPUT_DIR="${OUTPUT_DIR:-CAP/outputs_original/${DATASET}_bge_bs${BATCH_SIZE}_len${STREAM_LENGTH_BATCHES}_high}"
CACHE_DIR="${CACHE_DIR:-MMD_detection/cache}"

cmd=(
  python3 CAP/cap_original_detector.py
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --normal-source "${NORMAL_SOURCE}"
  --global-normal-path "${GLOBAL_NORMAL_PATH}"
  --embedding-model "${EMBEDDING_MODEL}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --stream-length-batches "${STREAM_LENGTH_BATCHES}"
  --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
  --calibration-streams "${CALIBRATION_STREAMS}"
  --threshold-percentile "${THRESHOLD_PERCENTILE}"
  --benign-eval-streams "${BENIGN_EVAL_STREAMS}"
  --attacker-eval-streams "${ATTACKER_EVAL_STREAMS}"
  --mixed-attacker-ratios "${MIXED_ATTACKER_RATIOS}"
  --mixed-streams-per-ratio "${MIXED_STREAMS_PER_RATIO}"
  --seed "${SEED}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --num-buckets "${NUM_BUCKETS}"
  --baseline-lambda "${BASELINE_LAMBDA}"
  --alpha "${ALPHA}"
  --beta "${BETA}"
  --coverage-weight "${COVERAGE_WEIGHT}"
  --novelty-weight "${NOVELTY_WEIGHT}"
  --spread-weight "${SPREAD_WEIGHT}"
  --spread-max "${SPREAD_MAX}"
)

echo "Running original-style CAP stream detection..."
echo "Dataset: ${DATASET}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Stream length batches: ${STREAM_LENGTH_BATCHES}"
echo "Tail: high"
echo "Output dir: ${OUTPUT_DIR}"
echo

"${cmd[@]}"
