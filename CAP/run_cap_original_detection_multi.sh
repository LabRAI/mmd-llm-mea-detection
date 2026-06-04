#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATASETS="${DATASETS:-model_leeching,Meaeq_AGNEWS,Meaeq_HATESPEECH,Meaeq_IMDB,Meaeq_SST-2,Query_efficent_med,ME_BERT_boolq_random,ME_BERT_boolq_wiki,ME_BERT_mnli_random,ME_BERT_mnli_wiki,ME_BERT_squad_random,ME_BERT_squad_wiki,ME_BERT_sst2_random,ME_BERT_sst2_wiki}"

BATCH_SIZE="${BATCH_SIZE:-100}"
STREAM_LENGTH_BATCHES="${STREAM_LENGTH_BATCHES:-10}"
DEVICE="${DEVICE:-cuda}"
CALIBRATION_STREAMS="${CALIBRATION_STREAMS:-1000}"
BENIGN_EVAL_STREAMS="${BENIGN_EVAL_STREAMS:-50}"
ATTACKER_EVAL_STREAMS="${ATTACKER_EVAL_STREAMS:-50}"
MIXED_STREAMS_PER_RATIO="${MIXED_STREAMS_PER_RATIO:-50}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
OUTPUT_ROOT="${OUTPUT_ROOT:-CAP/outputs_original}"

dataset_items="${DATASETS//,/ }"

echo "Running original-style CAP stream detection for multiple datasets..."
echo "Datasets: ${DATASETS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Stream length batches: ${STREAM_LENGTH_BATCHES}"
echo "Device: ${DEVICE}"
echo "Shared output root: ${OUTPUT_ROOT}"
echo

for dataset in ${dataset_items}; do
  if [[ -z "${dataset}" ]]; then
    continue
  fi

  dataset_output_dir="${OUTPUT_ROOT}/${dataset}_bge_bs${BATCH_SIZE}_len${STREAM_LENGTH_BATCHES}_high"

  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "Output dir: ${dataset_output_dir}"
  echo "============================================================"

  DATASET="${dataset}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  STREAM_LENGTH_BATCHES="${STREAM_LENGTH_BATCHES}" \
  DEVICE="${DEVICE}" \
  CALIBRATION_STREAMS="${CALIBRATION_STREAMS}" \
  BENIGN_EVAL_STREAMS="${BENIGN_EVAL_STREAMS}" \
  ATTACKER_EVAL_STREAMS="${ATTACKER_EVAL_STREAMS}" \
  MIXED_STREAMS_PER_RATIO="${MIXED_STREAMS_PER_RATIO}" \
  THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE}" \
  OUTPUT_DIR="${dataset_output_dir}" \
  bash "${SCRIPT_DIR}/run_cap_original_detection.sh"

  echo
done

echo "All requested original-style CAP detection runs finished."
