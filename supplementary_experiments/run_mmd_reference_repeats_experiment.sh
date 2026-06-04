#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DATASETS="${DATASETS:-model_leeching}"
SEEDS="${SEEDS:-42}"
REFERENCE_REPEATS_LIST="${REFERENCE_REPEATS_LIST:-1 5 10 20}"

BATCH_SIZE="${BATCH_SIZE:-1500}"
NULL_SAMPLES="${NULL_SAMPLES:-1000}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-0.05,0.1,0.25,0.5}"
MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO:-50}"
BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES:-50}"
ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES:-50}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
QUERY_PREFIX="${QUERY_PREFIX:-}"
DEVICE="${DEVICE:-cuda}"
MMD_DEVICE="${MMD_DEVICE:-cuda}"
MMD_DTYPE="${MMD_DTYPE:-float32}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-supplementary_experiments/mmd_reference_repeats_experiment}"
RESULT_ROOT="${RESULT_ROOT:-${EXPERIMENT_ROOT}/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/outputs}"
CACHE_ROOT="${CACHE_ROOT:-${EXPERIMENT_ROOT}/cache}"
LOG_ROOT="${LOG_ROOT:-${EXPERIMENT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE="${FORCE:-0}"
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-supplementary_experiments/shared_cache/mmd_embeddings}"
WARMUP_CACHE="${WARMUP_CACHE:-1}"

mkdir -p "${RESULT_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${LOG_ROOT}" "${SHARED_CACHE_DIR}"

SUMMARY_CSV="${SUMMARY_CSV:-${RESULT_ROOT}/mmd_reference_repeats_runs_bs${BATCH_SIZE}.csv}"
OVERALL_CSV="${OVERALL_CSV:-${RESULT_ROOT}/mmd_reference_repeats_overall_bs${BATCH_SIZE}.csv}"
printf "reference_repeats,dataset,seed,batch_size,null_samples,total_seconds,seconds_per_eval_unit,benign_fpr,5pct_tpr,10pct_tpr,25pct_tpr,50pct_tpr,100pct_tpr,avg_tpr,balanced_acc,output_dir,metadata_path,log_file,status\n" > "${SUMMARY_CSV}"

time_command() {
  local time_file="$1"
  shift
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

append_summary() {
  local reference_repeats="$1"
  local dataset="$2"
  local seed="$3"
  local output_dir="$4"
  local log_file="$5"
  local time_file="$6"
  local status="$7"
  "${PYTHON_BIN}" - <<PY >> "${SUMMARY_CSV}"
import csv
import json
import math
import sys
from pathlib import Path

reference_repeats = ${reference_repeats@Q}
dataset = ${dataset@Q}
seed = ${seed@Q}
batch_size = ${BATCH_SIZE@Q}
null_samples = ${NULL_SAMPLES@Q}
output_dir = ${output_dir@Q}
metadata_path = str(Path(output_dir) / "metadata.json")
log_file = ${log_file@Q}
time_file = Path(${time_file@Q})
status = ${status@Q}

total_seconds = float(time_file.read_text().strip()) if time_file.exists() and time_file.read_text().strip() else float("nan")
seconds_per_eval_unit = total_seconds / 300.0 if not math.isnan(total_seconds) else float("nan")

def rate(by_split, key):
    try:
        return float(by_split[key]["detection_rate"]) * 100.0
    except Exception:
        return float("nan")

row = {
    "reference_repeats": reference_repeats,
    "dataset": dataset,
    "seed": seed,
    "batch_size": batch_size,
    "null_samples": null_samples,
    "total_seconds": f"{total_seconds:.6f}" if not math.isnan(total_seconds) else "nan",
    "seconds_per_eval_unit": f"{seconds_per_eval_unit:.6f}" if not math.isnan(seconds_per_eval_unit) else "nan",
    "benign_fpr": "nan",
    "5pct_tpr": "nan",
    "10pct_tpr": "nan",
    "25pct_tpr": "nan",
    "50pct_tpr": "nan",
    "100pct_tpr": "nan",
    "avg_tpr": "nan",
    "balanced_acc": "nan",
    "output_dir": output_dir,
    "metadata_path": metadata_path,
    "log_file": log_file,
    "status": status,
}

path = Path(metadata_path)
if path.exists() and status == "0":
    meta = json.loads(path.read_text(encoding="utf-8"))
    by = meta.get("metrics_by_split", {})
    fpr = rate(by, "heldout_benign")
    tprs = {
        "5pct_tpr": rate(by, "mixed_0.05"),
        "10pct_tpr": rate(by, "mixed_0.1"),
        "25pct_tpr": rate(by, "mixed_0.25"),
        "50pct_tpr": rate(by, "mixed_0.5"),
        "100pct_tpr": rate(by, "attacker"),
    }
    vals = [v for v in tprs.values() if not math.isnan(v)]
    avg = sum(vals) / len(vals) if vals else float("nan")
    bal = (avg + 100.0 - fpr) / 2.0 if not math.isnan(avg) and not math.isnan(fpr) else float("nan")
    row["benign_fpr"] = f"{fpr:.6f}"
    row["avg_tpr"] = f"{avg:.6f}"
    row["balanced_acc"] = f"{bal:.6f}"
    row.update({k: f"{v:.6f}" for k, v in tprs.items()})

writer = csv.DictWriter(sys.stdout, fieldnames=[
    "reference_repeats", "dataset", "seed", "batch_size", "null_samples",
    "total_seconds", "seconds_per_eval_unit", "benign_fpr", "5pct_tpr",
    "10pct_tpr", "25pct_tpr", "50pct_tpr", "100pct_tpr", "avg_tpr",
    "balanced_acc", "output_dir", "metadata_path", "log_file", "status",
])
writer.writerow(row)
PY
}

write_overall() {
  "${PYTHON_BIN}" - <<PY
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

summary = Path(${SUMMARY_CSV@Q})
overall = Path(${OVERALL_CSV@Q})
metrics = ["total_seconds", "seconds_per_eval_unit", "benign_fpr", "5pct_tpr", "10pct_tpr", "25pct_tpr", "50pct_tpr", "100pct_tpr", "avg_tpr", "balanced_acc"]
rows = []
if summary.exists():
    with summary.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "0"]
groups = defaultdict(list)
for row in rows:
    groups[row["reference_repeats"]].append(row)
fields = ["reference_repeats", "runs"]
for metric in metrics:
    fields += [f"{metric}_mean", f"{metric}_std"]
with overall.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for key, group in sorted(groups.items(), key=lambda item: float(item[0])):
        out = {"reference_repeats": key, "runs": str(len(group))}
        for metric in metrics:
            vals = []
            for row in group:
                try:
                    val = float(row[metric])
                except Exception:
                    val = float("nan")
                if not math.isnan(val):
                    vals.append(val)
            out[f"{metric}_mean"] = f"{statistics.mean(vals):.6f}" if vals else "nan"
            out[f"{metric}_std"] = f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else ("0.000000" if vals else "nan")
        writer.writerow(out)
PY
}

dataset_items="${DATASETS//,/ }"

warmup_embedding_cache() {
  local dataset="$1"
  local seed="$2"
  local warmup_dir="${OUTPUT_ROOT}/warmup/${dataset}_seed${seed}_bs${BATCH_SIZE}"
  local warmup_log_dir="${LOG_ROOT}/warmup"
  local warmup_log="${warmup_log_dir}/${dataset}_seed${seed}_bs${BATCH_SIZE}.log"
  local marker="${SHARED_CACHE_DIR}/.warmup_${dataset}_seed${seed}_bs${BATCH_SIZE}_$(echo "${EMBEDDING_MODEL}_${QUERY_PREFIX}" | tr '/: ' '___')"

  if [[ "${WARMUP_CACHE}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${marker}" && "${FORCE}" != "1" ]]; then
    echo "Embedding cache warmup already marked for ${dataset}, seed ${seed}"
    return 0
  fi

  mkdir -p "${warmup_dir}" "${warmup_log_dir}"
  echo "Warming embedding cache for dataset=${dataset}, seed=${seed}"
  env DATASET="${dataset}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    REFERENCE_REPEATS="1" \
    NULL_SAMPLES="1" \
    THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE}" \
    NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO}" \
    MIXED_ATTACKER_RATIOS="" \
    MIXED_BATCHES_PER_RATIO="0" \
    BENIGN_EVAL_BATCHES="1" \
    ATTACKER_EVAL_BATCHES="1" \
    SEED="${seed}" \
    EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    QUERY_PREFIX="${QUERY_PREFIX}" \
    DEVICE="${DEVICE}" \
    MMD_DEVICE="${MMD_DEVICE}" \
    MMD_DTYPE="${MMD_DTYPE}" \
    CACHE_DIR="${SHARED_CACHE_DIR}" \
    OUTPUT_DIR="${warmup_dir}" \
    bash MMD_detection/run_mmd_detection_gpu.sh > "${warmup_log}" 2>&1
  touch "${marker}"
}

echo "Running MMD reference-repeats experiment..."
echo "Datasets: ${DATASETS}"
echo "Seeds: ${SEEDS}"
echo "Reference repeats: ${REFERENCE_REPEATS_LIST}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo

for reference_repeats in ${REFERENCE_REPEATS_LIST}; do
  for seed in ${SEEDS}; do
    for dataset in ${dataset_items}; do
      [[ -z "${dataset}" ]] && continue
      warmup_embedding_cache "${dataset}" "${seed}"
      run_name="ref${reference_repeats}/${dataset}_seed${seed}_bs${BATCH_SIZE}"
      output_dir="${OUTPUT_ROOT}/${run_name}"
      cache_dir="${SHARED_CACHE_DIR}"
      log_dir="${LOG_ROOT}/ref${reference_repeats}"
      log_file="${log_dir}/${dataset}_seed${seed}_bs${BATCH_SIZE}.log"
      time_file="${log_dir}/${dataset}_seed${seed}_bs${BATCH_SIZE}.time"
      metadata_path="${output_dir}/metadata.json"
      mkdir -p "${output_dir}" "${cache_dir}" "${log_dir}"
      if [[ "${FORCE}" != "1" && -f "${metadata_path}" && -f "${time_file}" ]]; then
        echo "Skipping existing run: ${run_name}"
        append_summary "${reference_repeats}" "${dataset}" "${seed}" "${output_dir}" "${log_file}" "${time_file}" "0"
        continue
      fi
      echo "============================================================"
      echo "Reference repeats: ${reference_repeats}"
      echo "Dataset: ${dataset}"
      echo "Seed: ${seed}"
      echo "Output dir: ${output_dir}"
      echo "============================================================"
      set +e
      time_command "${time_file}" env DATASET="${dataset}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        REFERENCE_REPEATS="${reference_repeats}" \
        NULL_SAMPLES="${NULL_SAMPLES}" \
        THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE}" \
        NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO}" \
        MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS}" \
        MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO}" \
        BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES}" \
        ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES}" \
        SEED="${seed}" \
        EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
        QUERY_PREFIX="${QUERY_PREFIX}" \
        DEVICE="${DEVICE}" \
        MMD_DEVICE="${MMD_DEVICE}" \
        MMD_DTYPE="${MMD_DTYPE}" \
        CACHE_DIR="${cache_dir}" \
        OUTPUT_DIR="${output_dir}" \
        bash MMD_detection/run_mmd_detection_gpu.sh > "${log_file}" 2>&1
      status="$?"
      set -e
      append_summary "${reference_repeats}" "${dataset}" "${seed}" "${output_dir}" "${log_file}" "${time_file}" "${status}"
      if [[ "${status}" != "0" ]]; then
        echo "Run failed: ${run_name}. See ${log_file}" >&2
        exit "${status}"
      fi
    done
  done
done

write_overall
echo
echo "Finished MMD reference-repeats experiment."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Overall CSV: ${OVERALL_CSV}"
