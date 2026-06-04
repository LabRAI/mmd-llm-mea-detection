#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

DEFAULT_DATASETS="model_leeching,Meaeq_AGNEWS,Meaeq_HATESPEECH,Meaeq_IMDB,Meaeq_SST-2,Query_efficent_med,ME_BERT_boolq_random,ME_BERT_boolq_wiki,ME_BERT_mnli_random,ME_BERT_mnli_wiki,ME_BERT_squad_random,ME_BERT_squad_wiki,ME_BERT_sst2_random,ME_BERT_sst2_wiki"
DATASETS="${DATASETS:-${DEFAULT_DATASETS}}"
SEEDS="${SEEDS:-42}"

# Format: model_name|query_prefix|short_slug
# Keep prefixes empty for ordinary SentenceTransformer/BGE models. E5-style
# query encoders are run with the recommended "query: " prefix.
MODEL_SPECS="${MODEL_SPECS:-BAAI/bge-small-en-v1.5||bge_small_en_v15;sentence-transformers/all-MiniLM-L6-v2||all_minilm_l6_v2;sentence-transformers/all-mpnet-base-v2||all_mpnet_base_v2;intfloat/e5-small-v2|query: |e5_small_v2;intfloat/e5-base-v2|query: |e5_base_v2}"

BATCH_SIZE="${BATCH_SIZE:-1500}"
REFERENCE_REPEATS="${REFERENCE_REPEATS:-20}"
NULL_SAMPLES="${NULL_SAMPLES:-1000}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95}"
NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO:-0.8}"
MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS:-0.05,0.1,0.25,0.5}"
MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO:-50}"
BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES:-50}"
ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES:-50}"

DEVICE="${DEVICE:-cuda}"
MMD_DEVICE="${MMD_DEVICE:-cuda}"
MMD_DTYPE="${MMD_DTYPE:-float32}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-supplementary_experiments/mmd_embedding_model_experiment}"
RESULT_ROOT="${RESULT_ROOT:-${EXPERIMENT_ROOT}/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXPERIMENT_ROOT}/outputs}"
CACHE_ROOT="${CACHE_ROOT:-${EXPERIMENT_ROOT}/cache}"
LOG_ROOT="${LOG_ROOT:-${EXPERIMENT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE="${FORCE:-0}"
PARALLEL_MODELS="${PARALLEL_MODELS:-1}"
MAX_MODEL_JOBS="${MAX_MODEL_JOBS:-0}"

mkdir -p "${RESULT_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${LOG_ROOT}"

SUMMARY_CSV="${SUMMARY_CSV:-${RESULT_ROOT}/mmd_embedding_model_runs_bs${BATCH_SIZE}.csv}"
OVERALL_CSV="${OVERALL_CSV:-${RESULT_ROOT}/mmd_embedding_model_overall_bs${BATCH_SIZE}.csv}"
BY_DATASET_CSV="${BY_DATASET_CSV:-${RESULT_ROOT}/mmd_embedding_model_by_dataset_bs${BATCH_SIZE}.csv}"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  printf "embedding_model,model_slug,query_prefix,dataset,seed,batch_size,benign_fpr,5pct_tpr,10pct_tpr,25pct_tpr,50pct_tpr,100pct_tpr,avg_tpr,balanced_acc,output_dir,metadata_path,log_file,status\n" > "${SUMMARY_CSV}"
fi

csv_escape() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

append_summary_from_metadata() {
  local embedding_model="$1"
  local model_slug="$2"
  local query_prefix="$3"
  local dataset="$4"
  local seed="$5"
  local output_dir="$6"
  local log_file="$7"
  local status="$8"
  local metadata_path="${output_dir}/metadata.json"

  local summary_line
  summary_line="$("${PYTHON_BIN}" - <<PY
import csv
import json
import math
import sys
from pathlib import Path

embedding_model = ${embedding_model@Q}
model_slug = ${model_slug@Q}
query_prefix = ${query_prefix@Q}
dataset = ${dataset@Q}
seed = ${seed@Q}
batch_size = ${BATCH_SIZE@Q}
output_dir = ${output_dir@Q}
metadata_path = ${metadata_path@Q}
log_file = ${log_file@Q}
status = ${status@Q}

def rate(metrics, key):
    try:
        value = metrics[key]["detection_rate"] * 100.0
    except Exception:
        value = float("nan")
    return value

row = {
    "embedding_model": embedding_model,
    "model_slug": model_slug,
    "query_prefix": query_prefix,
    "dataset": dataset,
    "seed": seed,
    "batch_size": batch_size,
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
    by_split = meta.get("metrics_by_split", {})
    fpr = rate(by_split, "heldout_benign")
    tprs = {
        "5pct_tpr": rate(by_split, "mixed_0.05"),
        "10pct_tpr": rate(by_split, "mixed_0.1"),
        "25pct_tpr": rate(by_split, "mixed_0.25"),
        "50pct_tpr": rate(by_split, "mixed_0.5"),
        "100pct_tpr": rate(by_split, "attacker"),
    }
    valid = [v for v in tprs.values() if not math.isnan(v)]
    avg_tpr = sum(valid) / len(valid) if valid else float("nan")
    balanced = (avg_tpr + 100.0 - fpr) / 2.0 if not math.isnan(avg_tpr) and not math.isnan(fpr) else float("nan")
    row.update({
        "benign_fpr": f"{fpr:.6f}",
        "avg_tpr": f"{avg_tpr:.6f}",
        "balanced_acc": f"{balanced:.6f}",
    })
    row.update({k: f"{v:.6f}" for k, v in tprs.items()})

writer = csv.DictWriter(sys.stdout, fieldnames=[
    "embedding_model", "model_slug", "query_prefix", "dataset", "seed",
    "batch_size", "benign_fpr", "5pct_tpr", "10pct_tpr", "25pct_tpr",
    "50pct_tpr", "100pct_tpr", "avg_tpr", "balanced_acc", "output_dir",
    "metadata_path", "log_file", "status",
])
writer.writerow(row)
PY
)"

  if command -v flock >/dev/null 2>&1; then
    {
      flock -x 200
      printf "%s\n" "${summary_line}"
    } 200>> "${SUMMARY_CSV}"
  else
    printf "%s\n" "${summary_line}" >> "${SUMMARY_CSV}"
  fi
}

write_aggregate_tables() {
  "${PYTHON_BIN}" - <<PY
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

summary_path = Path(${SUMMARY_CSV@Q})
overall_path = Path(${OVERALL_CSV@Q})
by_dataset_path = Path(${BY_DATASET_CSV@Q})

metrics = [
    "benign_fpr",
    "5pct_tpr",
    "10pct_tpr",
    "25pct_tpr",
    "50pct_tpr",
    "100pct_tpr",
    "avg_tpr",
    "balanced_acc",
]

def as_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")

rows = []
if summary_path.exists():
    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "0"]

def summarize(group_rows):
    out = {"runs": str(len(group_rows))}
    for metric in metrics:
        values = [as_float(r.get(metric, "nan")) for r in group_rows]
        values = [v for v in values if not math.isnan(v)]
        if values:
            out[f"{metric}_mean"] = f"{statistics.mean(values):.6f}"
            out[f"{metric}_std"] = f"{statistics.stdev(values):.6f}" if len(values) > 1 else "0.000000"
        else:
            out[f"{metric}_mean"] = "nan"
            out[f"{metric}_std"] = "nan"
    return out

overall_groups = defaultdict(list)
dataset_groups = defaultdict(list)
for row in rows:
    overall_groups[(row["embedding_model"], row["model_slug"], row["query_prefix"])].append(row)
    dataset_groups[(row["embedding_model"], row["model_slug"], row["query_prefix"], row["dataset"])].append(row)

overall_fields = ["embedding_model", "model_slug", "query_prefix", "runs"]
for metric in metrics:
    overall_fields.extend([f"{metric}_mean", f"{metric}_std"])

with overall_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=overall_fields)
    writer.writeheader()
    for (embedding_model, model_slug, query_prefix), group_rows in sorted(overall_groups.items()):
        row = {"embedding_model": embedding_model, "model_slug": model_slug, "query_prefix": query_prefix}
        row.update(summarize(group_rows))
        writer.writerow(row)

by_dataset_fields = ["embedding_model", "model_slug", "query_prefix", "dataset", "runs"]
for metric in metrics:
    by_dataset_fields.extend([f"{metric}_mean", f"{metric}_std"])

with by_dataset_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=by_dataset_fields)
    writer.writeheader()
    for (embedding_model, model_slug, query_prefix, dataset), group_rows in sorted(dataset_groups.items()):
        row = {
            "embedding_model": embedding_model,
            "model_slug": model_slug,
            "query_prefix": query_prefix,
            "dataset": dataset,
        }
        row.update(summarize(group_rows))
        writer.writerow(row)
PY
}

dataset_items="${DATASETS//,/ }"
model_specs="${MODEL_SPECS}"

echo "Running MMD embedding-model experiment..."
echo "Datasets: ${DATASETS}"
echo "Seeds: ${SEEDS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Parallel models: ${PARALLEL_MODELS}"
echo "Max model jobs: ${MAX_MODEL_JOBS}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo

IFS=';' read -r -a specs <<< "${model_specs}"

run_model_spec() {
  local spec="$1"
  [[ -z "${spec}" ]] && return 0
  local embedding_model query_prefix model_slug
  IFS='|' read -r embedding_model query_prefix model_slug <<< "${spec}"
  if [[ -z "${embedding_model}" || -z "${model_slug}" ]]; then
    echo "Invalid MODEL_SPECS entry: ${spec}" >&2
    exit 1
  fi

  echo "############################################################"
  echo "Embedding model: ${embedding_model}"
  echo "Model slug: ${model_slug}"
  echo "Query prefix: ${query_prefix:-<empty>}"
  echo "############################################################"

  for seed in ${SEEDS}; do
    for dataset in ${dataset_items}; do
      [[ -z "${dataset}" ]] && continue

      run_name="${model_slug}/${dataset}_seed${seed}_bs${BATCH_SIZE}"
      output_dir="${OUTPUT_ROOT}/${run_name}"
      cache_dir="${CACHE_ROOT}/${model_slug}"
      log_dir="${LOG_ROOT}/${model_slug}"
      log_file="${log_dir}/${dataset}_seed${seed}_bs${BATCH_SIZE}.log"
      metadata_path="${output_dir}/metadata.json"

      mkdir -p "${output_dir}" "${cache_dir}" "${log_dir}"

      if [[ "${FORCE}" != "1" && -f "${metadata_path}" ]]; then
        echo "Skipping existing run: ${run_name}"
        append_summary_from_metadata "${embedding_model}" "${model_slug}" "${query_prefix}" \
          "${dataset}" "${seed}" "${output_dir}" "${log_file}" "0"
        continue
      fi

      echo "============================================================"
      echo "Dataset: ${dataset}"
      echo "Seed: ${seed}"
      echo "Output dir: ${output_dir}"
      echo "Cache dir: ${cache_dir}"
      echo "Log file: ${log_file}"
      echo "============================================================"

      set +e
      env DATASET="${dataset}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        REFERENCE_REPEATS="${REFERENCE_REPEATS}" \
        NULL_SAMPLES="${NULL_SAMPLES}" \
        THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE}" \
        NORMAL_TRAIN_RATIO="${NORMAL_TRAIN_RATIO}" \
        MIXED_ATTACKER_RATIOS="${MIXED_ATTACKER_RATIOS}" \
        MIXED_BATCHES_PER_RATIO="${MIXED_BATCHES_PER_RATIO}" \
        BENIGN_EVAL_BATCHES="${BENIGN_EVAL_BATCHES}" \
        ATTACKER_EVAL_BATCHES="${ATTACKER_EVAL_BATCHES}" \
        SEED="${seed}" \
        EMBEDDING_MODEL="${embedding_model}" \
        QUERY_PREFIX="${query_prefix}" \
        DEVICE="${DEVICE}" \
        MMD_DEVICE="${MMD_DEVICE}" \
        MMD_DTYPE="${MMD_DTYPE}" \
        CACHE_DIR="${cache_dir}" \
        OUTPUT_DIR="${output_dir}" \
        bash MMD_detection/run_mmd_detection_gpu.sh > "${log_file}" 2>&1
      status="$?"
      set -e

      append_summary_from_metadata "${embedding_model}" "${model_slug}" "${query_prefix}" \
        "${dataset}" "${seed}" "${output_dir}" "${log_file}" "${status}"

      if [[ "${status}" != "0" ]]; then
        echo "Run failed: ${run_name}. See ${log_file}" >&2
        exit "${status}"
      fi
    done
  done
}

running_model_jobs() {
  jobs -rp | wc -l
}

wait_for_model_slot() {
  if [[ "${MAX_MODEL_JOBS}" -le 0 ]]; then
    return 0
  fi
  while [[ "$(running_model_jobs)" -ge "${MAX_MODEL_JOBS}" ]]; do
    sleep 5
  done
}

if [[ "${PARALLEL_MODELS}" == "1" ]]; then
  pids=()
  for spec in "${specs[@]}"; do
    [[ -z "${spec}" ]] && continue
    wait_for_model_slot
    run_model_spec "${spec}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "At least one embedding-model run failed. Check logs in ${LOG_ROOT}." >&2
    exit 1
  fi
else
  for spec in "${specs[@]}"; do
    run_model_spec "${spec}"
  done
fi

write_aggregate_tables

echo
echo "All MMD embedding-model runs finished."
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Overall CSV: ${OVERALL_CSV}"
echo "By-dataset CSV: ${BY_DATASET_CSV}"
