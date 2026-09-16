# UNSW-NB15 Deep Learning IDS

Research code for the UNSW-NB15 part of the thesis. This project studies deep learning for network intrusion detection under the official UNSW-NB15 train/test split, with emphasis on multiclass attack classification, minority-class behavior, and reproducible evaluation.

The main contribution of this branch is a feature-axis multi-scale CNN-BiLSTM-attention architecture evaluated against local deep-learning baselines, imbalance-aware variants, W&B hyperparameter sweeps, multi-seed verification, and operational metrics.

## Research Objective

UNSW-NB15 is not difficult mainly because of binary attack detection. The stronger research problem is multiclass classification under severe class imbalance. Large classes such as Normal, Generic, and Exploits dominate global accuracy, while minority classes such as Analysis, Backdoor, Shellcode, and Worms are much harder to detect.

This project therefore prioritizes:

- the official train/test split;
- preprocessing fitted only on training data;
- macro-F1 as the main model-selection signal;
- per-class precision, recall, F1, and false-positive rate;
- rare-class and difficult-class analysis;
- ablation of architecture and imbalance-handling choices;
- parameter count, model size, latency, and throughput.

## Dataset

Expected official files:

```text
data/raw/
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```

The dataset is not tracked by Git. Keep raw and processed data local.

## Project Layout

```text
.
  configs/                  Experiment configurations.
    sweeps/                 W&B sweep definitions.
  data/                     Local raw and processed data, ignored by Git.
  plan/                     Research plans, phase notes, and result summaries.
  reports/                  Human-readable progress reports and figures.
  runs/                     Local experiment outputs, ignored by Git.
  scripts/                  Setup checks, utilities, and W&B helpers.
  src/unsw_nb15_ids/        Installable Python package.
    models/                 CNN, LSTM, BiLSTM, CNN-LSTM, CNN-BiLSTM, attention models.
    config.py               TOML configuration loading.
    data.py                 Dataset loading, preprocessing, encoding, and splits.
    losses.py               Cross entropy, focal, balanced softmax, and EQL-style losses.
    metrics.py              Classification and operational metrics.
    train.py                Main training and evaluation pipeline.
  pyproject.toml            Python package metadata and CLI entry points.
  requirements.txt          CPU/general dependencies.
  requirements-gpu-cu124.txt CUDA 12.4 PyTorch dependencies.
```

## Proposed Architecture

The proposed model treats each encoded UNSW-NB15 flow as a sequence over feature positions instead of using a length-one pseudo-sequence. This allows sequence layers to learn interactions across feature dimensions.

Architecture:

1. Encoded tabular flow input.
2. Feature-axis sequence representation.
3. Multi-scale one-dimensional convolution using several kernel sizes.
4. Bidirectional LSTM encoder.
5. Multi-head self-attention.
6. Dense classification head.

The final selected model uses cross entropy with AdamW and label smoothing. SMOTENC, EQL-style loss, focal loss, and balanced softmax are kept as controlled ablation experiments, but they are not the final selected UNSW-NB15 configuration.

## Setup

Create a local environment from this project root:

```powershell
cd projects/unsw_nb15
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_setup.py
```

For NVIDIA GPU training with CUDA 12.4:

```powershell
python -m pip install --upgrade --force-reinstall -r requirements-gpu-cu124.txt
python -m pip install -e .
python scripts/check_setup.py --require-gpu
```

If the local machine uses another CUDA runtime, install PyTorch from the official selector, then rerun `python scripts/check_setup.py --require-gpu`.

## Reproducible Workflow

Validate the data and preprocessing:

```powershell
python scripts/check_setup.py
unsw-profile --config configs/multiclass.toml
unsw-validate-preprocessing --configs configs/binary.toml configs/multiclass.toml
```

Run core multiclass baselines:

```powershell
unsw-train --config configs/multiclass_cnn.toml
unsw-train --config configs/multiclass_lstm.toml
unsw-train --config configs/multiclass_bilstm.toml
unsw-train --config configs/multiclass_cnn_lstm.toml
unsw-train --config configs/multiclass_cnn_bilstm.toml
```

Run the proposed architecture and imbalance ablations:

```powershell
unsw-train --config configs/multiclass_feature_attention_ce.toml
unsw-train --config configs/multiclass_feature_attention_eql.toml
unsw-train --config configs/multiclass_feature_attention_smotenc_eql.toml
unsw-train --config configs/multiclass_feature_attention_cb_focal.toml
unsw-train --config configs/multiclass_feature_attention_balanced_softmax.toml
```

Aggregate multiclass results:

```powershell
unsw-multiclass-report --pattern "multiclass_*/metrics.json" --output runs/multiclass_comparison.csv --per-class-output runs/multiclass_per_class.csv --figures-dir runs/multiclass_figures
```

Run final fixed-seed verification:

```powershell
unsw-train --config configs/final_wandb_ce_seed42.toml
unsw-train --config configs/final_wandb_ce_seed7.toml
unsw-train --config configs/final_wandb_ce_seed123.toml
```

## W&B Sweeps

Sweep definitions live in:

```text
configs/sweeps/
```

Typical command pattern:

```powershell
wandb sweep --entity <entity> --project pfe-thesis-unsw-nb15 configs\sweeps\multiclass_feature_attention_ce.yaml
wandb agent <entity>/pfe-thesis-unsw-nb15/<sweep_id> --count 30
```

Use W&B for search and tracking only. Thesis claims should rely on rerunning the selected configuration with fixed seeds.

## Final Result Summary

Final selected model:

```text
W&B-tuned feature-axis multi-scale CNN-BiLSTM-attention classifier
Loss: cross entropy
Optimizer: AdamW
Selection: validation weighted-F1 with macro-F1-oriented analysis
Seeds: 42, 7, 123
```

Three-seed official test result:

| Metric | Mean +/- Std |
| --- | ---: |
| Accuracy | 75.77% +/- 1.36% |
| Macro-F1 | 47.27% +/- 1.17% |
| Weighted-F1 | 77.52% +/- 1.48% |
| Rare recall mean | 24.67% +/- 3.25% |
| Rare F1 mean | 19.30% +/- 1.60% |
| Difficult-class F1 mean | 20.92% +/- 2.25% |

Best observed seed:

| Seed | Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: |
| 42 | 77.34% | 48.59% | 79.23% |

The result should be framed as a strict-protocol macro-F1 and minority-behavior contribution, not as a highest-accuracy claim.

## Git Policy

Do not commit:

- raw or processed datasets;
- local virtual environments;
- local runs and W&B logs;
- trained checkpoints;
- generated cache files.

Keep source code, configs, research notes, and Markdown reports under version control.

## Suggested Repository Names

- `unsw-nb15-feature-axis-attention-ids`
- `unsw-nb15-deep-ids-benchmark`
- `unsw-nb15-multiclass-dl-ids`

Recommended: `unsw-nb15-feature-axis-attention-ids`
