# MMD-LLM-MEA-Detection

Official implementation and processed query-traffic artifact for:

**An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic**

This repository provides code for benign-calibrated detection of LLM model extraction attacks in API query traffic.
The main detector applies maximum mean discrepancy (MMD) to sentence embeddings and compares incoming traffic windows against benign reference traffic.
The artifact also includes adapted and original-protocol baselines used in the paper: PRADA, SEAT, CAP, DATE, and marginal Mahalanobis distance.

## Overview

The repository supports three types of experiments:

- **Proposed detector:** benign-calibrated MMD over semantic query embeddings.
- **Unified adapted baselines:** PRADA, SEAT, CAP, DATE, and Mahalanobis evaluated under a shared traffic-window protocol.
- **Original-protocol baselines:** baseline-specific protocols used to study whether prior methods transfer directly to text-query model extraction traffic.

The default setting uses `BAAI/bge-small-en-v1.5` as the sentence embedding model and evaluates both pure attacker traffic and mixed benign-attacker traffic windows.

## Repository Structure

```text
MMD_detection/              # proposed benign-calibrated MMD detector
PRADA/                      # PRADA-style adapted and stream detectors
SEAT/                       # SEAT-style adapted and original account detectors
CAP/                        # CAP-style coverage detector
DATE/                       # DATE-style text anomaly detector
Mahalanobis/                # marginal Mahalanobis detector
data/                       # processed query files used by the experiments
supplementary_experiments/  # extra MMD sensitivity and runtime scripts
requirements.txt            # shared Python dependencies
```

## Installation

Install the shared dependencies from the repository root:

```bash
pip install -r requirements.txt
```

GPU execution is recommended for embedding, MMD, and nearest-neighbor computations.

## Quick Start

Run the proposed MMD detector on one dataset:

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

## Data Preparation

Each dataset is stored in JSONL format:

```text
data/<dataset>/normal/queries.jsonl
data/<dataset>/attacker/queries.jsonl
```

Each row contains a `query` field:

```json
{"id": "example-0", "index": 0, "query": "What is the capital of France?"}
```

Two MNLI normal-query files exceed GitHub's 100MB single-file limit, so they are stored as split parts:

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

## Reproducing Experiments

### Proposed MMD Detector

| Experiment | Command |
| --- | --- |
| Single dataset | `bash MMD_detection/run_mmd_detection_gpu.sh` |
| Main multi-dataset run | `bash MMD_detection/run_mmd_detection_multi_gpu.sh` |
| Batch-size and threshold sensitivity | `bash MMD_detection/run_mmd_sensitivity_gpu.sh` |

The MMD detector writes `metadata.json` and `batch_scores.csv` under the selected output directory.

### Adapted Baselines

| Method | Command |
| --- | --- |
| PRADA | `bash PRADA/run_prada_detection.sh` |
| SEAT | `bash SEAT/run_seat_detection.sh` |
| CAP | `bash CAP/run_cap_detection.sh` |
| DATE | `bash DATE/run_date_detection.sh` |
| Mahalanobis | `bash Mahalanobis/run_mahalanobis_detection.sh` |

### Original-Protocol Baselines

| Method | Command |
| --- | --- |
| PRADA | `bash PRADA/run_prada_stream_detection.sh` |
| SEAT | `bash SEAT/run_seat_original_detection.sh` |
| CAP | `bash CAP/run_cap_original_detection.sh` |
| DATE | `bash DATE/run_date_original_detection.sh` |
| Mahalanobis | `bash Mahalanobis/run_mahalanobis_sample_detection.sh` |

Multi-dataset wrappers are available when the corresponding folder provides a `*_multi.sh` script.

### Supplementary Experiments

The `supplementary_experiments/` folder contains scripts for extra MMD sensitivity and runtime analyses:

```bash
bash supplementary_experiments/run_mmd_embedding_model_experiment.sh
bash supplementary_experiments/run_mmd_reference_repeats_experiment.sh
bash supplementary_experiments/run_mmd_null_samples_experiment.sh
bash supplementary_experiments/run_cold_cache_wallclock.sh
```

By default, these scripts write aggregate CSV files under `supplementary_experiments/<experiment-name>/results/`.
They may also create local `logs/`, `outputs/`, and `cache/` directories during execution.

## Configuration

Most scripts accept the following environment variables:

```bash
DATASET=model_leeching
DATASETS=model_leeching,Query_efficent_med,Meaeq_HATESPEECH
BATCH_SIZE=1500
SEED=42
DEVICE=cuda
OUTPUT_ROOT=<output-directory>
```

Several adapted baselines can be run with either one-sided or two-sided benign-calibrated decisions:

```bash
TAIL=upper bash PRADA/run_prada_detection.sh
TAIL=two-sided bash PRADA/run_prada_detection.sh

DETECTION_TAIL=upper bash SEAT/run_seat_detection.sh
DETECTION_TAIL=two-sided bash SEAT/run_seat_detection.sh

TAIL=high bash CAP/run_cap_detection.sh
TAIL=two-sided bash CAP/run_cap_detection.sh

TEST_SIDEDNESS=upper bash DATE/run_date_detection.sh
TEST_SIDEDNESS=two-sided bash DATE/run_date_detection.sh
```

The Mahalanobis baseline uses the marginal distance score with an upper-tail benign-calibrated threshold.

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

## Citation

If you find this repository useful, please cite the paper. A BibTeX entry will be added after the public preprint is available.

## Contact

For questions about the code or experiments, please open an issue in this repository.
