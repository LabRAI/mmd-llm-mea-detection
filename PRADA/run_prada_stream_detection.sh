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
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-${DEVICE}}"

STREAM_LENGTH="${STREAM_LENGTH:-3000}"
CALIBRATION_STREAMS="${CALIBRATION_STREAMS:-100}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
DISTANCE_METRIC="${DISTANCE_METRIC:-l2}"
MIN_DISTANCES="${MIN_DISTANCES:-100}"
MAX_SHAPIRO_SAMPLES="${MAX_SHAPIRO_SAMPLES:-5000}"
SHAPIRO_INTERVAL="${SHAPIRO_INTERVAL:-1}"
BENIGN_EVAL_STREAMS="${BENIGN_EVAL_STREAMS:-50}"
ATTACKER_EVAL_STREAMS="${ATTACKER_EVAL_STREAMS:-50}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-}"
MIXED_STREAMS_PER_RATIO="${MIXED_STREAMS_PER_RATIO:-50}"
MAX_NORMAL="${MAX_NORMAL:-}"
MAX_ATTACKER="${MAX_ATTACKER:-}"
SEED="${SEED:-42}"

OUTPUT_ROOT="${OUTPUT_ROOT:-PRADA/stream_outputs}"
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
    dataset_output_dir="${OUTPUT_ROOT}/${dataset}_bge_stream${STREAM_LENGTH}"
  fi

  cmd=(
    "${PYTHON_BIN}"
    PRADA/prada_stream_detector.py
    --dataset "${dataset}"
    --data-root "${DATA_ROOT}"
    --normal-source "${NORMAL_SOURCE}"
    --global-normal-path "${GLOBAL_NORMAL_PATH}"
    --embedding-model "${EMBEDDING_MODEL}"
    --stream-length "${STREAM_LENGTH}"
    --calibration-streams "${CALIBRATION_STREAMS}"
    --threshold-percentile "${THRESHOLD_PERCENTILE}"
    --normal-train-ratio "${NORMAL_TRAIN_RATIO}"
    --distance-metric "${DISTANCE_METRIC}"
    --compute-device "${COMPUTE_DEVICE}"
    --min-distances "${MIN_DISTANCES}"
    --max-shapiro-samples "${MAX_SHAPIRO_SAMPLES}"
    --shapiro-interval "${SHAPIRO_INTERVAL}"
    --benign-eval-streams "${BENIGN_EVAL_STREAMS}"
    --attacker-eval-streams "${ATTACKER_EVAL_STREAMS}"
    --mixed-attacker-ratios "${MIXED_ATTACKER_RATIOS}"
    --mixed-streams-per-ratio "${MIXED_STREAMS_PER_RATIO}"
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
  if [[ "${SEQUENTIAL_EVAL:-0}" == "1" ]]; then
    cmd+=(--sequential-eval)
  fi
  if [[ "${KEEP_LAST_STREAM:-0}" == "1" ]]; then
    cmd+=(--keep-last-stream)
  fi
  if [[ "${NO_PRECOMPUTE_DISTANCES:-0}" == "1" ]]; then
    cmd+=(--no-precompute-distances)
  fi
  if [[ "${NO_STOP_ON_DETECTION:-0}" == "1" ]]; then
    cmd+=(--no-stop-on-detection)
  fi

  echo "Running stream-level PRADA-style detection..."
  echo "Dataset: ${dataset}"
  echo "Normal source: ${NORMAL_SOURCE}"
  echo "Embedding model: ${EMBEDDING_MODEL}"
  echo "Stream length: ${STREAM_LENGTH}"
  echo "Shapiro interval: ${SHAPIRO_INTERVAL}"
  echo "Mixed attacker ratios: ${MIXED_ATTACKER_RATIOS:-none}"
  echo "Distance metric: ${DISTANCE_METRIC}"
  echo "Compute device: ${COMPUTE_DEVICE}"
  echo "Python: ${PYTHON_BIN}"
  echo "Output dir: ${dataset_output_dir}"
  echo

  "${cmd[@]}"
done
