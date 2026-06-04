#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET="${DATASET:-model_leeching}"
DATA_ROOT="${DATA_ROOT:-data}"
NORMAL_SOURCE="${NORMAL_SOURCE:-dataset}"
# NORMAL_SOURCE="${NORMAL_SOURCE:-global}"
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

BATCH_SIZE="${BATCH_SIZE:-1500}"
NULL_SAMPLES="${NULL_SAMPLES:-1000}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
TEST_SIDEDNESS="${TEST_SIDEDNESS:-two-sided}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-0.05,0.1,0.25,0.5}"
MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO:-50}"
BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES:-50}"
ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES:-50}"
BATCH_SCORE_AGG="${BATCH_SCORE_AGG:-mean}"
SCORE_QUANTILE="${SCORE_QUANTILE:-0.9}"
TOPK_FRACTION="${TOPK_FRACTION:-0.2}"
SEED="${SEED:-42}"

OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-date_${GENERATOR}_bs${BATCH_SIZE}}"
OUTPUT_DIR="${OUTPUT_DIR:-DATE/outputs/${DATASET}_${OUTPUT_SUFFIX}}"
SAVE_MODEL_DIR="${SAVE_MODEL_DIR:-}"

cmd=(
  python3 DATE/date_detector.py
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
  --batch-size "${BATCH_SIZE}"
  --null-samples "${NULL_SAMPLES}"
  --threshold-percentile "${THRESHOLD_PERCENTILE}"
  --test-sidedness "${TEST_SIDEDNESS}"
  --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
  --mixed-attacker-ratios "${MIXED_ATTACKER_RATIOS}"
  --mixed-batches-per-ratio "${MIXED_BATCHES_PER_RATIO}"
  --benign-eval-batches "${BENIGN_EVAL_BATCHES}"
  --attacker-eval-batches "${ATTACKER_EVAL_BATCHES}"
  --batch-score-agg "${BATCH_SCORE_AGG}"
  --score-quantile "${SCORE_QUANTILE}"
  --topk-fraction "${TOPK_FRACTION}"
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

if [[ -n "${SAVE_MODEL_DIR}" ]]; then
  cmd+=(--save-model-dir "${SAVE_MODEL_DIR}")
fi

echo "Running DATE detection..."
echo "Dataset: ${DATASET}"
echo "Normal source: ${NORMAL_SOURCE}"
echo "Model: ${MODEL_NAME}"
echo "Generator: ${GENERATOR}"
if [[ "${GENERATOR}" == "learned" ]]; then
  echo "Generator model: ${GENERATOR_MODEL_NAME}"
fi
echo "Max train steps: ${MAX_TRAIN_STEPS}"
echo "Test sidedness: ${TEST_SIDEDNESS}"
if [[ -n "${MAX_NORMAL}" ]]; then
  echo "Max normal queries: ${MAX_NORMAL}"
fi
if [[ -n "${MAX_ATTACKER}" ]]; then
  echo "Max attacker queries: ${MAX_ATTACKER}"
fi
echo "Batch aggregation: ${BATCH_SCORE_AGG}"
echo "Output dir: ${OUTPUT_DIR}"
echo

"${cmd[@]}"
