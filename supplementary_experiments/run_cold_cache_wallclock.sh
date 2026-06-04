#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET="${DATASET:-model_leeching}"
BATCH_SIZE="${BATCH_SIZE:-1500}"
SEED="${SEED:-42}"
REPEATS="${REPEATS:-1}"
METHODS="${METHODS:-MMD Mahalanobis PRADA SEAT CAP DATE}"
PARALLEL="${PARALLEL:-1}"
MAX_JOBS="${MAX_JOBS:-0}"
DEVICE="${DEVICE:-cuda}"
MMD_DEVICE="${MMD_DEVICE:-${DEVICE}}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-${DEVICE}}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
QUERY_PREFIX="${QUERY_PREFIX:-}"
MMD_REFERENCE_REPEATS="${MMD_REFERENCE_REPEATS:-5}"
MMD_NULL_SAMPLES="${MMD_NULL_SAMPLES:-250}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-supplementary_experiments/cold_cache_wallclock}"
RESULT_ROOT="${RESULT_ROOT:-${EXPERIMENT_ROOT}/results}"
TMP_CACHE_ROOT="${TMP_CACHE_ROOT:-${EXPERIMENT_ROOT}/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/outputs}"
LOG_ROOT="${LOG_ROOT:-${EXPERIMENT_ROOT}/logs}"
KEEP_CACHE="${KEEP_CACHE:-0}"
DATE_CHECKPOINT_WARMUP="${DATE_CHECKPOINT_WARMUP:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATE_MODEL_NAME="${DATE_MODEL_NAME:-google/electra-small-discriminator}"
DATE_GENERATOR="${DATE_GENERATOR:-learned}"
DATE_GENERATOR_MODEL_NAME="${DATE_GENERATOR_MODEL_NAME:-google/electra-small-generator}"

mkdir -p "${RESULT_ROOT}" "${TMP_CACHE_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}"

RESULT_CSV="${RESULT_CSV:-${RESULT_ROOT}/cold_cache_wallclock_${DATASET}_bs${BATCH_SIZE}_seed${SEED}.csv}"
RESULT_ROWS_DIR="${RESULT_ROWS_DIR:-${RESULT_ROOT}/rows_${DATASET}_bs${BATCH_SIZE}_seed${SEED}}"

TIME_BIN="${TIME_BIN:-}"
if [[ -z "${TIME_BIN}" ]]; then
  if [[ -x "/usr/bin/time" ]]; then
    TIME_BIN="/usr/bin/time"
  elif command -v gtime >/dev/null 2>&1; then
    TIME_BIN="$(command -v gtime)"
  elif command -v time >/dev/null 2>&1; then
    TIME_BIN="$(command -v time)"
  fi
fi

rm -rf "${RESULT_ROWS_DIR}"
mkdir -p "${RESULT_ROWS_DIR}"
printf "method,dataset,seed,repeat,batch_size,cold_cache,total_seconds,num_eval_units,seconds_per_eval_unit,cache_dir,output_dir,log_file,status,notes\n" > "${RESULT_CSV}"

csv_escape() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

timed_run() {
  local time_file="$1"
  shift

  if [[ -n "${TIME_BIN}" && -x "${TIME_BIN}" ]]; then
    "${TIME_BIN}" -f "%e" -o "${time_file}" "$@"
    return "$?"
  fi

  local start end status
  start="$("${PYTHON_BIN}" - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
)"
  "$@"
  status="$?"
  end="$("${PYTHON_BIN}" - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
)"
  "${PYTHON_BIN}" - <<PY > "${time_file}"
start = float("${start}")
end = float("${end}")
print(f"{end - start:.6f}")
PY
  return "${status}"
}

append_result() {
  local method="$1"
  local repeat="$2"
  local total_seconds="$3"
  local eval_units="$4"
  local cache_dir="$5"
  local output_dir="$6"
  local log_file="$7"
  local status="$8"
  local notes="$9"

  local seconds_per_unit
  seconds_per_unit="$("${PYTHON_BIN}" - <<PY
total = float("${total_seconds}")
units = int("${eval_units}")
print("nan" if units <= 0 else f"{total / units:.6f}")
PY
)"

  local result_line
  result_line="$(
    {
    csv_escape "${method}"; printf ","
    csv_escape "${DATASET}"; printf ","
    csv_escape "${SEED}"; printf ","
    csv_escape "${repeat}"; printf ","
    csv_escape "${BATCH_SIZE}"; printf ","
    csv_escape "true"; printf ","
    csv_escape "${total_seconds}"; printf ","
    csv_escape "${eval_units}"; printf ","
    csv_escape "${seconds_per_unit}"; printf ","
    csv_escape "${cache_dir}"; printf ","
    csv_escape "${output_dir}"; printf ","
    csv_escape "${log_file}"; printf ","
    csv_escape "${status}"; printf ","
    csv_escape "${notes}"; printf "\n"
    }
  )"

  local row_file="${RESULT_ROWS_DIR}/rep${repeat}_${method}.csv"
  printf "%s\n" "${result_line}" > "${row_file}"
}

combine_results() {
  printf "method,dataset,seed,repeat,batch_size,cold_cache,total_seconds,num_eval_units,seconds_per_eval_unit,cache_dir,output_dir,log_file,status,notes\n" > "${RESULT_CSV}"
  if compgen -G "${RESULT_ROWS_DIR}/*.csv" >/dev/null; then
    cat "${RESULT_ROWS_DIR}"/*.csv >> "${RESULT_CSV}"
  fi
}

ensure_date_checkpoints() {
  if [[ "${DATE_CHECKPOINT_WARMUP}" != "1" ]]; then
    return 0
  fi
  if [[ " ${METHODS} " != *" DATE "* ]]; then
    return 0
  fi

  echo "Checking DATE HuggingFace checkpoints before timed runs..."
  "${PYTHON_BIN}" - <<PY
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

model_name = "${DATE_MODEL_NAME}"
generator = "${DATE_GENERATOR}"
generator_model_name = "${DATE_GENERATOR_MODEL_NAME}"

def ensure_tokenizer_and_model(name, model_cls=None):
    try:
        AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=True)
        if model_cls is not None:
            model_cls.from_pretrained(name, local_files_only=True)
        print(f"[local] {name}")
    except Exception:
        print(f"[download] {name}")
        AutoTokenizer.from_pretrained(name, use_fast=True)
        if model_cls is not None:
            model_cls.from_pretrained(name)

ensure_tokenizer_and_model(model_name, AutoModel)
if generator == "learned":
    ensure_tokenizer_and_model(generator_model_name, AutoModelForMaskedLM)
PY
}

method_eval_units() {
  local method="$1"
  case "${method}" in
    MMD|Mahalanobis|PRADA|SEAT|DATE)
      echo 300
      ;;
    CAP)
      echo 300
      ;;
    *)
      echo 0
      ;;
  esac
}

run_method() {
  local method="$1"
  local repeat="$2"
  local method_lc
  method_lc="$(echo "${method}" | tr '[:upper:]' '[:lower:]')"

  local run_id="${method_lc}_${DATASET}_bs${BATCH_SIZE}_seed${SEED}_rep${repeat}"
  local cache_dir="${TMP_CACHE_ROOT}/${run_id}"
  local output_dir="${OUTPUT_ROOT}/${run_id}"
  local log_file="${LOG_ROOT}/${run_id}.log"
  local time_file="${LOG_ROOT}/${run_id}.time"
  local eval_units
  eval_units="$(method_eval_units "${method}")"

  rm -rf "${cache_dir}" "${output_dir}"
  mkdir -p "${cache_dir}" "${output_dir}"

  echo "============================================================"
  echo "Method: ${method}"
  echo "Dataset: ${DATASET}"
  echo "Repeat: ${repeat}/${REPEATS}"
  echo "Cache dir: ${cache_dir}"
  echo "Output dir: ${output_dir}"
  echo "Log file: ${log_file}"
  if [[ "${method}" != "DATE" ]]; then
    echo "Embedding model: ${EMBEDDING_MODEL}"
  fi
  if [[ "${method}" == "MMD" ]]; then
    echo "MMD reference repeats: ${MMD_REFERENCE_REPEATS}"
    echo "MMD null samples: ${MMD_NULL_SAMPLES}"
  fi
  echo "============================================================"

  set +e
  case "${method}" in
    MMD)
      timed_run "${time_file}" \
        env DATASET="${DATASET}" BATCH_SIZE="${BATCH_SIZE}" SEED="${SEED}" \
          DEVICE="${DEVICE}" MMD_DEVICE="${MMD_DEVICE}" \
          EMBEDDING_MODEL="${EMBEDDING_MODEL}" QUERY_PREFIX="${QUERY_PREFIX}" \
          REFERENCE_REPEATS="${MMD_REFERENCE_REPEATS}" NULL_SAMPLES="${MMD_NULL_SAMPLES}" \
          CACHE_DIR="${cache_dir}" OUTPUT_DIR="${output_dir}" \
          bash MMD_detection/run_mmd_detection_gpu.sh > "${log_file}" 2>&1
      ;;
    Mahalanobis)
      timed_run "${time_file}" \
        env DATASET="${DATASET}" BATCH_SIZE="${BATCH_SIZE}" SEED="${SEED}" \
          DEVICE="${DEVICE}" EMBEDDING_MODEL="${EMBEDDING_MODEL}" QUERY_PREFIX="${QUERY_PREFIX}" \
          CACHE_DIR="${cache_dir}" OUTPUT_DIR="${output_dir}" \
          bash Mahalanobis/run_mahalanobis_detection.sh > "${log_file}" 2>&1
      ;;
    PRADA)
      timed_run "${time_file}" \
        env DATASET="${DATASET}" BATCH_SIZE="${BATCH_SIZE}" SEED="${SEED}" \
          DEVICE="${DEVICE}" COMPUTE_DEVICE="${COMPUTE_DEVICE}" \
          EMBEDDING_MODEL="${EMBEDDING_MODEL}" QUERY_PREFIX="${QUERY_PREFIX}" \
          CACHE_DIR="${cache_dir}" OUTPUT_DIR="${output_dir}" \
          bash PRADA/run_prada_detection.sh > "${log_file}" 2>&1
      ;;
    SEAT)
      timed_run "${time_file}" \
        env DATASET="${DATASET}" BATCH_SIZE="${BATCH_SIZE}" SEED="${SEED}" \
          DEVICE="${DEVICE}" COMPUTE_DEVICE="${COMPUTE_DEVICE}" \
          EMBEDDING_MODEL="${EMBEDDING_MODEL}" QUERY_PREFIX="${QUERY_PREFIX}" \
          CACHE_DIR="${cache_dir}" OUTPUT_DIR="${output_dir}" \
          bash SEAT/run_seat_detection.sh > "${log_file}" 2>&1
      ;;
    CAP)
      timed_run "${time_file}" \
        env DATASET="${DATASET}" BATCH_SIZE="${BATCH_SIZE}" SEED="${SEED}" \
          DEVICE="${DEVICE}" EMBEDDING_MODEL="${EMBEDDING_MODEL}" QUERY_PREFIX="${QUERY_PREFIX}" \
          CACHE_DIR="${cache_dir}" OUTPUT_DIR="${output_dir}" \
          bash CAP/run_cap_detection.sh > "${log_file}" 2>&1
      ;;
    DATE)
      rm -rf "${cache_dir}"
      timed_run "${time_file}" \
        env DATASET="${DATASET}" BATCH_SIZE="${BATCH_SIZE}" SEED="${SEED}" \
          DEVICE="${DEVICE}" OUTPUT_DIR="${output_dir}" \
          MODEL_NAME="${DATE_MODEL_NAME}" GENERATOR="${DATE_GENERATOR}" \
          GENERATOR_MODEL_NAME="${DATE_GENERATOR_MODEL_NAME}" \
          bash DATE/run_date_detection.sh > "${log_file}" 2>&1
      ;;
    *)
      echo "Unknown method: ${method}" > "${log_file}"
      false
      ;;
  esac
  local status="$?"
  set -e

  local total_seconds="nan"
  if [[ -s "${time_file}" ]]; then
    total_seconds="$(tail -n 1 "${time_file}")"
  fi

  local notes="cold embedding cache; embedding_model=${EMBEDDING_MODEL}; full existing run script"
  if [[ "${method}" == "MMD" ]]; then
    notes="cold embedding cache; embedding_model=${EMBEDDING_MODEL}; reference_repeats=${MMD_REFERENCE_REPEATS}; null_samples=${MMD_NULL_SAMPLES}; full existing run script"
  fi
  if [[ "${method}" == "DATE" ]]; then
    notes="full DATE run; checkpoint download warmed before timing; includes DATE training"
    cache_dir="N/A"
  fi

  append_result "${method}" "${repeat}" "${total_seconds}" "${eval_units}" \
    "${cache_dir}" "${output_dir}" "${log_file}" "${status}" "${notes}"

  if [[ "${KEEP_CACHE}" != "1" && "${method}" != "DATE" ]]; then
    rm -rf "${cache_dir}"
  fi

  if [[ "${status}" != "0" ]]; then
    echo "Method ${method} failed with status ${status}. See ${log_file}" >&2
    return "${status}"
  fi
}

ensure_date_checkpoints

running_jobs() {
  jobs -rp | wc -l
}

wait_for_slot() {
  if [[ "${MAX_JOBS}" -le 0 ]]; then
    return 0
  fi
  while [[ "$(running_jobs)" -ge "${MAX_JOBS}" ]]; do
    sleep 2
  done
}

for repeat in $(seq 1 "${REPEATS}"); do
  if [[ "${PARALLEL}" == "1" ]]; then
    pids=()
    for method in ${METHODS}; do
      wait_for_slot
      run_method "${method}" "${repeat}" &
      pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if [[ "${failed}" != "0" ]]; then
      combine_results
      echo "At least one method failed during repeat ${repeat}. Check logs in ${LOG_ROOT}." >&2
      exit 1
    fi
  else
    for method in ${METHODS}; do
      run_method "${method}" "${repeat}"
    done
  fi
done

combine_results

echo
echo "Timing results written to ${RESULT_CSV}"
