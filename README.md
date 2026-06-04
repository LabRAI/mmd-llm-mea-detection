# Model Extraction Query-Traffic Detection

This repository contains the code and processed query files for evaluating benign-calibrated query-traffic detection against model extraction attacks.
The main detector uses maximum mean discrepancy (MMD) over sentence embeddings and compares incoming traffic windows against benign reference traffic.
The repository also includes adapted and original-protocol baselines used in the paper: PRADA, SEAT, CAP, DATE, and marginal Mahalanobis distance.

## Repository Structure

```text
MMD_detection/    # proposed benign-calibrated MMD detector
PRADA/            # PRADA-style adapted and stream detectors
SEAT/             # SEAT-style adapted and original account detectors
CAP/              # CAP-style coverage detector
DATE/             # DATE-style text anomaly detector
Mahalanobis/      # marginal Mahalanobis detector
data/             # processed query files used by the experiments
requirements.txt  # shared Python dependencies
```

## Installation

Install the shared dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The experiments use `BAAI/bge-small-en-v1.5` as the default sentence embedding model.
GPU execution is recommended for embedding and MMD/nearest-neighbor computations.

## Data Format

Each dataset is stored in JSONL format:

```text
data/<dataset>/normal/queries.jsonl
data/<dataset>/attacker/queries.jsonl
```

Each row contains a `query` field:

```json
{"id": "example-0", "index": 0, "query": "What is the capital of France?"}
```

Two MNLI normal-query files exceed GitHub's 100MB single-file limit, so they are stored as split parts in this artifact:

```text
data/ME_BERT_mnli_random/normal_parts/
data/ME_BERT_mnli_wiki/normal_parts/
```

Before running experiments on the two MNLI pairs, reconstruct the expected JSONL files:

```bash
mkdir -p data/ME_BERT_mnli_random/normal data/ME_BERT_mnli_wiki/normal
cat data/ME_BERT_mnli_random/normal_parts/queries_*.part > data/ME_BERT_mnli_random/normal/queries.jsonl
cat data/ME_BERT_mnli_wiki/normal_parts/queries_*.part > data/ME_BERT_mnli_wiki/normal/queries.jsonl
```

## Main MMD Detector

Run one dataset:

```bash
DATASET=model_leeching \
BATCH_SIZE=1500 \
DEVICE=cuda \
MMD_DEVICE=cuda \
SEED=42 \
bash MMD_detection/run_mmd_detection_gpu.sh
```

Run the default fourteen attacker-normal pairs:

```bash
bash MMD_detection/run_mmd_detection_multi_gpu.sh
```

Run the three main seeds:

```bash
SEED=1 bash MMD_detection/run_mmd_detection_multi_gpu.sh
SEED=20 bash MMD_detection/run_mmd_detection_multi_gpu.sh
SEED=42 bash MMD_detection/run_mmd_detection_multi_gpu.sh
```

Run batch-size and threshold sensitivity experiments:

```bash
bash MMD_detection/run_mmd_sensitivity_gpu.sh
```

The MMD detector writes `metadata.json` and `batch_scores.csv` under the selected output directory.

## Supplementary Experiments

The `supplementary_experiments/` folder contains scripts for extra MMD sensitivity and runtime analyses:

```bash
bash supplementary_experiments/run_mmd_embedding_model_experiment.sh
bash supplementary_experiments/run_mmd_reference_repeats_experiment.sh
bash supplementary_experiments/run_mmd_null_samples_experiment.sh
bash supplementary_experiments/run_cold_cache_wallclock.sh
```

By default, these scripts write aggregate CSV files under `supplementary_experiments/<experiment-name>/results/`.
They may also create local `logs/`, `outputs/`, and `cache/` directories during execution.

## Adapted Baselines

Run adapted PRADA:

```bash
bash PRADA/run_prada_detection.sh
```

Run adapted SEAT:

```bash
bash SEAT/run_seat_detection.sh
```

Run adapted CAP:

```bash
bash CAP/run_cap_detection.sh
```

Run adapted DATE:

```bash
bash DATE/run_date_detection.sh
```

Run adapted marginal Mahalanobis:

```bash
bash Mahalanobis/run_mahalanobis_detection.sh
```

Most scripts accept shared environment variables:

```bash
DATASET=model_leeching
DATASETS=model_leeching,Query_efficent_med,Meaeq_HATESPEECH
BATCH_SIZE=1500
DEVICE=cuda
SEED=42
OUTPUT_ROOT=<output-directory>
```

## One-Sided and Two-Sided Decisions

Several adapted baselines can be run with either one-sided or two-sided benign-calibrated decisions.
The one-sided setting follows the expected suspicious direction of the original method.
The two-sided setting flags scores that fall outside the benign calibration interval.

PRADA uses `TAIL`:

```bash
TAIL=upper bash PRADA/run_prada_detection.sh      # one-sided
TAIL=two-sided bash PRADA/run_prada_detection.sh  # two-sided
```

SEAT uses `DETECTION_TAIL`:

```bash
DETECTION_TAIL=upper bash SEAT/run_seat_detection.sh      # one-sided
DETECTION_TAIL=two-sided bash SEAT/run_seat_detection.sh  # two-sided
```

CAP uses `TAIL` with CAP's original high-coverage direction:

```bash
TAIL=high bash CAP/run_cap_detection.sh        # one-sided
TAIL=two-sided bash CAP/run_cap_detection.sh   # two-sided
```

DATE uses `TEST_SIDEDNESS`:

```bash
TEST_SIDEDNESS=upper bash DATE/run_date_detection.sh      # one-sided
TEST_SIDEDNESS=two-sided bash DATE/run_date_detection.sh  # two-sided
```

The Mahalanobis baseline in this artifact uses the marginal distance score with an upper-tail benign-calibrated threshold.

## Original-Protocol Baselines

The paper also evaluates whether original baseline protocols transfer to this setting.
Use the following scripts for the original-style runs:

```bash
bash PRADA/run_prada_stream_detection.sh
bash SEAT/run_seat_original_detection.sh
bash CAP/run_cap_original_detection.sh
bash DATE/run_date_original_detection.sh
bash Mahalanobis/run_mahalanobis_sample_detection.sh
```

Multi-dataset wrappers are available when the corresponding folder provides a `*_multi.sh` script.

## Method Notes

The adapted protocol uses the same processed query datasets and the same traffic-window construction across methods.
Benign queries are split into a benign reference/calibration pool and a held-out benign evaluation pool.
Attacker queries are used to construct pure attacker windows and mixed windows with attacker fractions such as `0.05`, `0.10`, `0.25`, and `0.50`.

MMD calibrates a benign-only null distribution using benign-vs-benign comparisons.
PRADA scores nearest-neighbor distance normality.
SEAT scores similar-pair behavior.
CAP scores embedding-space coverage.
DATE scores text anomalies using self-supervised transformer objectives.
Mahalanobis scores distance from an estimated benign embedding distribution.

## Outputs

Most detectors write:

```text
metadata.json
batch_scores.csv
```

Original query-level detectors may instead write:

```text
query_scores.csv
metadata.json
```

These outputs are sufficient to compute benign FPR, attacker TPR, mixed-traffic TPR, average TPR, and balanced accuracy.

## Anonymity Note

This artifact is prepared for anonymous review.
It intentionally excludes paper drafts, local caches, generated embedding files, and unrelated intermediate files.
