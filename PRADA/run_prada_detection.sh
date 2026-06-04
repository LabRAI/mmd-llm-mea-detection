#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET="${DATASET:-model_leeching,Meaeq_AGNEWS,Meaeq_HATESPEECH,Meaeq_IMDB,Meaeq_SST-2,Query_efficent_med,ME_BERT_boolq_random,ME_BERT_boolq_wiki,ME_BERT_mnli_random,ME_BERT_mnli_wiki,ME_BERT_squad_random,ME_BERT_squad_wiki,ME_BERT_sst2_random,ME_BERT_sst2_wiki}"
DATA_ROOT="${DATA_ROOT:-data}"
NORMAL_SOURCE="${NORMAL_SOURCE:-dataset}"
GLOBAL_NORMAL_PATH="${GLOBAL_NORMAL_PATH:-data/global_normal/queries.jsonl}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
DEVICE="${DEVICE:-cuda}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-${DEVICE}}"

BATCH_SIZE="${BATCH_SIZE:-1500}"
NULL_SAMPLES="${NULL_SAMPLES:-1000}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
DISTANCE_METRIC="${DISTANCE_METRIC:-l2}"
TAIL="${TAIL:-upper}"
MAX_SHAPIRO_SAMPLES="${MAX_SHAPIRO_SAMPLES:-5000}"
MAX_NORMAL="${MAX_NORMAL:-}"
MAX_ATTACKER="${MAX_ATTACKER:-}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-0.05,0.1,0.25,0.5}"
MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO:-50}"
BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES:-50}"
ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES:-50}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUTPUT_ROOT="${OUTPUT_ROOT:-PRADA/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
CACHE_DIR="${CACHE_DIR:-PRADA/cache}"

IFS=',' read -r -a DATASETS <<< "${DATASET}"

for raw_dataset in "${DATASETS[@]}"; do
  dataset="$(echo "${raw_dataset}" | xargs)"
  if [[ -z "${dataset}" ]]; then
    continue
  fi

  if [[ -n "${OUTPUT_DIR}" && "${#DATASETS[@]}" -eq 1 ]]; then
    dataset_output_dir="${OUTPUT_DIR}"
  else
    dataset_output_dir="${OUTPUT_ROOT}/${dataset}_bge_bs${BATCH_SIZE}"
  fi

  cmd=(
    "${PYTHON_BIN}"
    PRADA/prada_detector.py
    --dataset "${dataset}"
    --data-root "${DATA_ROOT}"
    --normal-source "${NORMAL_SOURCE}"
    --global-normal-path "${GLOBAL_NORMAL_PATH}"
    --embedding-model "${EMBEDDING_MODEL}"
    --batch-size "${BATCH_SIZE}"
    --null-samples "${NULL_SAMPLES}"
    --threshold-percentile "${THRESHOLD_PERCENTILE}"
    --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
    --distance-metric "${DISTANCE_METRIC}"
    --compute-device "${COMPUTE_DEVICE}"
    --tail "${TAIL}"
    --max-shapiro-samples "${MAX_SHAPIRO_SAMPLES}"
    --mixed-attacker-ratios "${MIXED_ATTACKER_RATIOS}"
    --mixed-batches-per-ratio "${MIXED_BATCHES_PER_RATIO}"
    --benign-eval-batches "${BENIGN_EVAL_BATCHES}"
    --attacker-eval-batches "${ATTACKER_EVAL_BATCHES}"
    --seed "${SEED}"
    --cache-dir "${CACHE_DIR}"
    --output-dir "${dataset_output_dir}"
  )

  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi
  if [[ -n "${MAX_NORMAL}" ]]; then
    cmd+=(--max-normal "${MAX_NORMAL}")
  fi
  if [[ -n "${MAX_ATTACKER}" ]]; then
    cmd+=(--max-attacker "${MAX_ATTACKER}")
  fi

  echo "Running batch PRADA-inspired detection..."
  echo "Dataset: ${dataset}"
  echo "Normal source: ${NORMAL_SOURCE}"
  echo "Embedding model: ${EMBEDDING_MODEL}"
  echo "Distance metric: ${DISTANCE_METRIC}"
  echo "Compute device: ${COMPUTE_DEVICE}"
  echo "Tail: ${TAIL}"
  echo "Python: ${PYTHON_BIN}"
  echo "Output dir: ${dataset_output_dir}"
  echo

  "${cmd[@]}"
done
