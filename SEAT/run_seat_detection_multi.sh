#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Comma-separated or whitespace-separated dataset names.
DEFAULT_DATASETS="model_leeching,Meaeq_AGNEWS,Meaeq_HATESPEECH,Meaeq_IMDB,Meaeq_SST-2,Query_efficent_med,ME_BERT_boolq_random,ME_BERT_boolq_wiki,ME_BERT_mnli_random,ME_BERT_mnli_wiki,ME_BERT_squad_random,ME_BERT_squad_wiki,ME_BERT_sst2_random,ME_BERT_sst2_wiki"
DATASETS="${DATASETS:-${DEFAULT_DATASETS}}"

BATCH_SIZE="${BATCH_SIZE:-1500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-SEAT/outputs}"

dataset_items="${DATASETS:-${DEFAULT_DATASETS}}"
dataset_items="${dataset_items//,/ }"

echo "Running SEAT-style detection for multiple datasets..."
echo "Datasets: ${DATASETS}"
echo "Shared output root: ${OUTPUT_ROOT}"
echo

for dataset in ${dataset_items}; do
  if [[ -z "${dataset}" ]]; then
    continue
  fi

  dataset_output_dir="${OUTPUT_ROOT}/${dataset}_bge_bs${BATCH_SIZE}"

  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "Output dir: ${dataset_output_dir}"
  echo "============================================================"

  DATASET="${dataset}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  OUTPUT_DIR="${dataset_output_dir}" \
  bash "${SCRIPT_DIR}/run_seat_detection.sh"

  echo
done

echo "All requested SEAT detection runs finished."
