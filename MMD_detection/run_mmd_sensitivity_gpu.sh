#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Representative datasets used for MMD sensitivity experiments.
SENSITIVITY_DATASETS="${SENSITIVITY_DATASETS:-model_leeching,Query_efficent_med,Meaeq_HATESPEECH,ME_BERT_squad_wiki,ME_BERT_squad_random}"

# Sensitivity grids. Comma-separated or whitespace-separated values are both accepted.
BATCH_SIZES="${BATCH_SIZES:-100 200 500 1000 1500}"
THRESHOLD_PERCENTILES="${THRESHOLD_PERCENTILES:-90 95 97.5 99}"

# Threshold sensitivity uses one fixed batch size.
BATCH_SIZE_FOR_THRESHOLD="${BATCH_SIZE_FOR_THRESHOLD:-1500}"

# Shared output root for all sensitivity runs.
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-experiment_results/MMD_gpu}"

# Optional switches.
RUN_BATCH_SIZE_SENSITIVITY="${RUN_BATCH_SIZE_SENSITIVITY:-1}"
RUN_THRESHOLD_SENSITIVITY="${RUN_THRESHOLD_SENSITIVITY:-1}"

batch_size_items="${BATCH_SIZES//,/ }"
threshold_items="${THRESHOLD_PERCENTILES//,/ }"

echo "Running GPU MMD sensitivity experiments..."
echo "Datasets: ${SENSITIVITY_DATASETS}"
echo "Output root base: ${OUTPUT_ROOT_BASE}"
echo

if [[ "${RUN_BATCH_SIZE_SENSITIVITY}" == "1" ]]; then
  echo "============================================================"
  echo "MMD batch size sensitivity"
  echo "Batch sizes: ${BATCH_SIZES}"
  echo "============================================================"

  for bs in ${batch_size_items}; do
    if [[ -z "${bs}" ]]; then
      continue
    fi

    echo
    echo ">>> Running batch_size=${bs}"
    DATASETS="${SENSITIVITY_DATASETS}" \
    BATCH_SIZE="${bs}" \
    OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/batch_size_bs${bs}" \
    bash "${SCRIPT_DIR}/run_mmd_detection_multi_gpu.sh"
  done
fi

if [[ "${RUN_THRESHOLD_SENSITIVITY}" == "1" ]]; then
  echo
  echo "============================================================"
  echo "MMD threshold sensitivity"
  echo "Threshold percentiles: ${THRESHOLD_PERCENTILES}"
  echo "Fixed batch size: ${BATCH_SIZE_FOR_THRESHOLD}"
  echo "============================================================"

  for th in ${threshold_items}; do
    if [[ -z "${th}" ]]; then
      continue
    fi

    echo
    echo ">>> Running threshold_percentile=${th}"
    DATASETS="${SENSITIVITY_DATASETS}" \
    BATCH_SIZE="${BATCH_SIZE_FOR_THRESHOLD}" \
    THRESHOLD_PERCENTILE="${th}" \
    OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/threshold_${th}" \
    bash "${SCRIPT_DIR}/run_mmd_detection_multi_gpu.sh"
  done
fi

echo
echo "All requested GPU MMD sensitivity experiments finished."
