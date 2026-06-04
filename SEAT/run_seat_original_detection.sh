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

ACCOUNT_SIZE="${ACCOUNT_SIZE:-1000}"
SIMILAR_PAIR_THRESHOLD="${SIMILAR_PAIR_THRESHOLD:-50}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-}"
CALIBRATION_MODE="${CALIBRATION_MODE:-pair-percentile}"
TARGET_FPR="${TARGET_FPR:-0.0001}"
CALIBRATION_ACCOUNTS="${CALIBRATION_ACCOUNTS:-50}"
BINARY_SEARCH_STEPS="${BINARY_SEARCH_STEPS:-16}"
PAIR_SAMPLE_SIZE="${PAIR_SAMPLE_SIZE:-200000}"
PAIR_TAIL_SCALE="${PAIR_TAIL_SCALE:-0.25}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-}"
MIXED_ACCOUNTS_PER_RATIO="${MIXED_ACCOUNTS_PER_RATIO:-50}"
BENIGN_EVAL_ACCOUNTS="${BENIGN_EVAL_ACCOUNTS:-50}"
ATTACKER_EVAL_ACCOUNTS="${ATTACKER_EVAL_ACCOUNTS:-50}"
SEQUENTIAL_EVAL="${SEQUENTIAL_EVAL:-0}"
SEED="${SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-SEAT/original_outputs/${DATASET}_bge_as${ACCOUNT_SIZE}}"
CACHE_DIR="${CACHE_DIR:-SEAT/cache}"

cmd=(
  python3 SEAT/seat_original_detector.py
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --normal-source "${NORMAL_SOURCE}"
  --global-normal-path "${GLOBAL_NORMAL_PATH}"
  --embedding-model "${EMBEDDING_MODEL}"
  --device "${DEVICE}"
  --compute-device "${COMPUTE_DEVICE}"
  --account-size "${ACCOUNT_SIZE}"
  --similar-pair-threshold "${SIMILAR_PAIR_THRESHOLD}"
  --calibration-mode "${CALIBRATION_MODE}"
  --target-fpr "${TARGET_FPR}"
  --calibration-accounts "${CALIBRATION_ACCOUNTS}"
  --binary-search-steps "${BINARY_SEARCH_STEPS}"
  --pair-sample-size "${PAIR_SAMPLE_SIZE}"
  --pair-tail-scale "${PAIR_TAIL_SCALE}"
  --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
  --mixed-attacker-ratios "${MIXED_ATTACKER_RATIOS}"
  --mixed-accounts-per-ratio "${MIXED_ACCOUNTS_PER_RATIO}"
  --benign-eval-accounts "${BENIGN_EVAL_ACCOUNTS}"
  --attacker-eval-accounts "${ATTACKER_EVAL_ACCOUNTS}"
  --seed "${SEED}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${SIMILARITY_THRESHOLD}" ]]; then
  cmd+=(--similarity-threshold "${SIMILARITY_THRESHOLD}")
fi

if [[ "${SEQUENTIAL_EVAL}" == "1" ]]; then
  cmd+=(--sequential-eval)
fi

echo "Running original SEAT-style account-level detection..."
echo "Dataset: ${DATASET}"
echo "Normal source: ${NORMAL_SOURCE}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Device: ${DEVICE}"
echo "Compute device: ${COMPUTE_DEVICE}"
echo "Account size: ${ACCOUNT_SIZE}"
echo "N_thresh: ${SIMILAR_PAIR_THRESHOLD}"
echo "Calibration mode: ${CALIBRATION_MODE}"
echo "Output dir: ${OUTPUT_DIR}"
echo

"${cmd[@]}"
