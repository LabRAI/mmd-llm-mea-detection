#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET="${DATASET:-model_leeching}"
DATA_ROOT="${DATA_ROOT:-data}"
NORMAL_SOURCE="${NORMAL_SOURCE:-dataset}"
GLOBAL_NORMAL_PATH="${GLOBAL_NORMAL_PATH:-data/global_normal/queries.jsonl}"
DEVICE="${DEVICE:-cuda}"
MAX_NORMAL="${MAX_NORMAL:-}"
MAX_ATTACKER="${MAX_ATTACKER:-}"

MODEL_NAME="${MODEL_NAME:-google/electra-small-discriminator}"
GENERATOR="${GENERATOR:-learned}"
GENERATOR_MODEL_NAME="${GENERATOR_MODEL_NAME:-google/electra-small-generator}"
MAX_LENGTH="${MAX_LENGTH:-128}"
MASK_PATTERNS="${MASK_PATTERNS:-50}"
MASK_RATIO="${MASK_RATIO:-0.5}"
EPOCHS="${EPOCHS:-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
RTD_WEIGHT="${RTD_WEIGHT:-50}"
RMD_WEIGHT="${RMD_WEIGHT:-100}"

THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
TEST_SIDEDNESS="${TEST_SIDEDNESS:-two-sided}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
BENIGN_EVAL_LIMIT="${BENIGN_EVAL_LIMIT:-}"
ATTACKER_EVAL_LIMIT="${ATTACKER_EVAL_LIMIT:-}"
INCLUDE_TEXT="${INCLUDE_TEXT:-0}"
SEED="${SEED:-42}"

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-date_original_${GENERATOR}}"
OUTPUT_DIR="${OUTPUT_DIR:-DATE/original_outputs/${DATASET}_${OUTPUT_SUFFIX}}"
SAVE_MODEL_DIR="${SAVE_MODEL_DIR:-}"

cmd=(
  python3 DATE/date_original_detector.py
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --normal-source "${NORMAL_SOURCE}"
  --global-normal-path "${GLOBAL_NORMAL_PATH}"
  --model-name "${MODEL_NAME}"
  --generator "${GENERATOR}"
  --max-length "${MAX_LENGTH}"
  --mask-patterns "${MASK_PATTERNS}"
  --mask-ratio "${MASK_RATIO}"
  --epochs "${EPOCHS}"
  --max-train-steps "${MAX_TRAIN_STEPS}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --rtd-weight "${RTD_WEIGHT}"
  --rmd-weight "${RMD_WEIGHT}"
  --threshold-percentile "${THRESHOLD_PERCENTILE}"
  --test-sidedness "${TEST_SIDEDNESS}"
  --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MAX_NORMAL}" ]]; then
  cmd+=(--max-normal "${MAX_NORMAL}")
fi

if [[ -n "${MAX_ATTACKER}" ]]; then
  cmd+=(--max-attacker "${MAX_ATTACKER}")
fi

if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi

if [[ "${GENERATOR}" == "learned" && -n "${GENERATOR_MODEL_NAME}" ]]; then
  cmd+=(--generator-model-name "${GENERATOR_MODEL_NAME}")
fi

if [[ -n "${BENIGN_EVAL_LIMIT}" ]]; then
  cmd+=(--benign-eval-limit "${BENIGN_EVAL_LIMIT}")
fi

if [[ -n "${ATTACKER_EVAL_LIMIT}" ]]; then
  cmd+=(--attacker-eval-limit "${ATTACKER_EVAL_LIMIT}")
fi

if [[ "${INCLUDE_TEXT}" == "1" ]]; then
  cmd+=(--include-text)
fi

if [[ -n "${SAVE_MODEL_DIR}" ]]; then
  cmd+=(--save-model-dir "${SAVE_MODEL_DIR}")
fi

echo "Running original DATE query-level detection..."
echo "Dataset: ${DATASET}"
echo "Normal source: ${NORMAL_SOURCE}"
echo "Model: ${MODEL_NAME}"
echo "Generator: ${GENERATOR}"
if [[ "${GENERATOR}" == "learned" ]]; then
  echo "Generator model: ${GENERATOR_MODEL_NAME}"
fi
echo "Max train steps: ${MAX_TRAIN_STEPS}"
echo "Test sidedness: ${TEST_SIDEDNESS}"
echo "Output dir: ${OUTPUT_DIR}"
echo

"${cmd[@]}"
