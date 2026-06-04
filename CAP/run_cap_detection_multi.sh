#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Comma-separated or whitespace-separated dataset names.
# Examples:
#   DATASETS="model_leeching,Meaeq,Query_efficent_med"
#   DATASETS="Stealing_Part Towards_More_Realistic_Extraction_Attacks"
DATASETS="${DATASETS:-model_leeching,Meaeq_AGNEWS,Meaeq_HATESPEECH,Meaeq_IMDB,Meaeq_SST-2,Query_efficent_med,ME_BERT_boolq_random,ME_BERT_boolq_wiki,ME_BERT_mnli_random,ME_BERT_mnli_wiki,ME_BERT_squad_random,ME_BERT_squad_wiki,ME_BERT_sst2_random,ME_BERT_sst2_wiki}"

# Shared experiment parameters. These are forwarded to run_cap_detection.sh.
# Defaults are chosen for batch-level comparison against MMD_detection.
BATCH_SIZE="${BATCH_SIZE:-1500}"
DEVICE="${DEVICE:-cuda}"
GROUP_MODE="${GROUP_MODE:-stream}"
STREAM_LENGTH_BATCHES="${STREAM_LENGTH_BATCHES:-1}"
CALIBRATION_STREAMS="${CALIBRATION_STREAMS:-1000}"
BENIGN_EVAL_STREAMS="${BENIGN_EVAL_STREAMS:-50}"
ATTACKER_EVAL_STREAMS="${ATTACKER_EVAL_STREAMS:-50}"
MIXED_STREAMS_PER_RATIO="${MIXED_STREAMS_PER_RATIO:-50}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
TAIL="${TAIL:-high}"

# Shared output root. Each dataset writes to:
#   ${OUTPUT_ROOT}/${DATASET}_bge_bs${BATCH_SIZE}_cap_${GROUP_MODE}_len${STREAM_LENGTH_BATCHES}_${TAIL}
OUTPUT_ROOT="${OUTPUT_ROOT:-CAP/outputs_batch_compare}"

dataset_items="${DATASETS//,/ }"

echo "Running CAP detection for multiple datasets..."
echo "Datasets: ${DATASETS}"
echo "Group mode: ${GROUP_MODE}"
echo "Stream length batches: ${STREAM_LENGTH_BATCHES}"
echo "Tail: ${TAIL}"
echo "Shared output root: ${OUTPUT_ROOT}"
echo

for dataset in ${dataset_items}; do
  if [[ -z "${dataset}" ]]; then
    continue
  fi

  dataset_output_dir="${OUTPUT_ROOT}/${dataset}_bge_bs${BATCH_SIZE}_cap_${GROUP_MODE}_len${STREAM_LENGTH_BATCHES}_${TAIL}"

  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "Output dir: ${dataset_output_dir}"
  echo "============================================================"

  DATASET="${dataset}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  DEVICE="${DEVICE}" \
  GROUP_MODE="${GROUP_MODE}" \
  STREAM_LENGTH_BATCHES="${STREAM_LENGTH_BATCHES}" \
  CALIBRATION_STREAMS="${CALIBRATION_STREAMS}" \
  BENIGN_EVAL_STREAMS="${BENIGN_EVAL_STREAMS}" \
  ATTACKER_EVAL_STREAMS="${ATTACKER_EVAL_STREAMS}" \
  MIXED_STREAMS_PER_RATIO="${MIXED_STREAMS_PER_RATIO}" \
  THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE}" \
  TAIL="${TAIL}" \
  OUTPUT_DIR="${dataset_output_dir}" \
  bash "${SCRIPT_DIR}/run_cap_detection.sh"

  echo
done

echo "All requested CAP detection runs finished."
