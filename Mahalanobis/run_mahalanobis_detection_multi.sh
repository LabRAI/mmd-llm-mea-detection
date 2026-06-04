#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Comma-separated or whitespace-separated dataset names.
# Examples:
#   DATASETS="model_leeching,Meaeq_AGNEWS,Meaeq_HATESPEECH"
#   DATASETS="Stealing_Part Towards_More_Realistic_Extraction_Attacks"
DEFAULT_DATASETS="model_leeching,Meaeq_AGNEWS,Meaeq_HATESPEECH,Meaeq_IMDB,Meaeq_SST-2,Query_efficent_med,ME_BERT_boolq_random,ME_BERT_boolq_wiki,ME_BERT_mnli_random,ME_BERT_mnli_wiki,ME_BERT_squad_random,ME_BERT_squad_wiki,ME_BERT_sst2_random,ME_BERT_sst2_wiki"
DATASETS="${DATASETS:-${DEFAULT_DATASETS}}"

# Shared experiment parameters. These are forwarded to run_mahalanobis_detection.sh.
BATCH_SIZE="${BATCH_SIZE:-1500}"

# Shared output root. Each dataset writes to:
#   ${OUTPUT_ROOT}/${DATASET}_bge_bs${BATCH_SIZE}
OUTPUT_ROOT="${OUTPUT_ROOT:-Mahalanobis/outputs}"

dataset_items="${DATASETS:-${DEFAULT_DATASETS}}"
dataset_items="${dataset_items//,/ }"

echo "Running marginal Mahalanobis detection for multiple datasets..."
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
  bash "${SCRIPT_DIR}/run_mahalanobis_detection.sh"

  echo
done

echo "All requested marginal Mahalanobis detection runs finished."
