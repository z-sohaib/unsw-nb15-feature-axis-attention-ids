---
title: "UNSW-NB15 Progress Report"
author: "Thesis Progress Report"
date: "2026-08-07"
---

# UNSW-NB15 Progress Report

## 1. Objective

This report summarizes the UNSW-NB15 experimental branch of the thesis. The
main objective was to move beyond binary intrusion detection and focus on the
harder multiclass setting, where global accuracy can hide failure on rare
attack categories.

The main gap addressed is:

> Global accuracy is not enough for UNSW-NB15 multiclass intrusion detection,
> because a model can perform well on majority classes while still failing on
> rare attack categories such as Analysis, Backdoor, Shellcode, and Worms.

## 2. Dataset And Protocol

UNSW-NB15 is a network intrusion detection dataset created at UNSW Canberra.
The project used the official train/test split.

| Split | Records | Usage |
| --- | ---: | --- |
| Official training set | 175,341 | Training plus validation split |
| Official test set | 82,332 | Final untouched evaluation |

Protocol safeguards:

- validation split created only from the official training set;
- test set kept untouched until final evaluation;
- encoders and scalers fitted only on training data;
- categorical features encoded: `proto`, `service`, `state`;
- numerical features transformed with `log1p` and scaling;
- SMOTENC applied only to training data when enabled;
- final results reported using accuracy, macro-F1, weighted-F1, per-class
  precision/recall/F1, false-positive rates, confusion matrices, latency,
  throughput, parameter count, and multi-seed stability.

## 3. Work Completed

| Phase | Work Completed | Status |
| --- | --- | --- |
| Phase 1 | Project setup, GPU/dependency checks, dataset profile | Done |
| Phase 2 | Strict preprocessing, split validation, saved preprocessors | Done |
| Phase 3 | Binary DL baselines to validate the pipeline | Done |
| Phase 4 | Multiclass baselines, proposed model, imbalance experiments | Done |
| Phase 5 | Final evaluation, W&B search, multi-seed verification | Done |

Binary classification was used as a pipeline validation step. The best binary
model reached 90.98% accuracy and 90.75% macro-F1. The thesis contribution is
the multiclass work.

## 4. Proposed Architecture

The proposed model is a feature-axis multi-scale CNN-BiLSTM-attention
architecture.

Instead of treating a tabular flow record as a length-1 pseudo-sequence, the
model treats encoded feature positions as sequence steps. This allows Conv1D,
BiLSTM, and attention layers to learn interactions across the feature vector.

Architecture:

```text
Encoded tabular feature vector
  -> feature-axis sequence representation
  -> parallel Conv1D kernels
  -> concatenate multi-scale feature maps
  -> BiLSTM encoder
  -> multi-head self-attention
  -> dense multiclass classifier
```

Final W&B-tuned configuration:

| Component | Final Setting |
| --- | --- |
| Conv1D channels | 48 |
| Kernel sizes | 3, 7, 9 |
| BiLSTM hidden size | 160 |
| BiLSTM layers | 2 |
| Attention heads | 2 |
| Dense hidden size | 96 |
| Dropout | 0.3900 |
| Optimizer | AdamW |
| Loss | Cross entropy |
| Label smoothing | 0.02 |
| SMOTENC | Disabled |
| Checkpoint selection | Validation weighted-F1 |

EQL v2-style loss and SMOTENC were also implemented and evaluated as
imbalance-handling ablations, but they were not selected as the final model.

## 5. Main Results

The final selected model is:

> W&B-tuned feature-axis multi-scale CNN-BiLSTM-attention trained with cross
> entropy, AdamW, label smoothing, and no SMOTENC.

| Model | Accuracy | Macro-F1 | Weighted-F1 | Difficult-Class Mean F1 | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| CNN | 73.85% | 39.77% | 74.38% | 7.57% | Lightweight baseline |
| LSTM | 77.18% | 46.25% | 77.36% | 18.13% | Best original accuracy baseline |
| BiLSTM | 74.79% | 45.94% | 75.56% | 18.20% | Recurrent baseline |
| CNN-LSTM | 76.20% | 44.32% | 76.52% | 15.48% | Hybrid baseline |
| CNN-BiLSTM | 75.04% | 44.49% | 75.95% | 15.92% | Hybrid baseline |
| Feature Attention + CE | 76.78% | 47.12% | 77.97% | 19.94% | Best pre-W&B CE model |
| Feature Attention + EQL v2 only | 72.18% | 47.45% | 75.61% | 22.83% | Best pre-W&B difficult-class model |
| Feature Attention + SMOTENC + EQL v2 | 72.79% | 46.28% | 76.75% | 19.62% | Rare-recall diagnostic model |
| **Final W&B Feature Attention + CE** | **75.77% +/- 1.36%** | **47.27% +/- 1.17%** | **77.52% +/- 1.48%** | **20.92% +/- 2.25%** | **Final selected model** |

The W&B-tuned CE model was selected because it gave the best overall balance of
macro-F1, accuracy, weighted-F1, false-positive risk, and multi-seed stability.

## 6. Final Multi-Seed Verification

| Run | Accuracy | Macro-F1 | Weighted-F1 | Rare F1 Mean | Difficult F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 42 | 77.34% | 48.59% | 79.23% | 18.27% | 23.46% |
| Seed 7 | 74.86% | 46.39% | 76.69% | 18.48% | 19.22% |
| Seed 123 | 75.11% | 46.82% | 76.63% | 21.14% | 20.07% |
| **Mean +/- std** | **75.77% +/- 1.36%** | **47.27% +/- 1.17%** | **77.52% +/- 1.48%** | **19.30% +/- 1.60%** | **20.92% +/- 2.25%** |

The seed-42 result is the best observed run, but the three-seed mean is the
valid thesis result.

## 7. External Comparison

The closest verified like-for-like comparison is the Wiley FMB-IDM paper, which
reports UNSW-NB15 multiclass results under the official train/test split.

| Criterion | Project Result | Wiley FMB-IDM | Outcome |
| --- | ---: | ---: | --- |
| Accuracy | 75.77% +/- 1.36%; best seed 77.34% | 78.87% | Wiley higher |
| F1 / Macro-F1-equivalent | 47.27% +/- 1.17%; best seed 48.59% | 46.08% | Project higher |

Interpretation:

- The project does not claim the highest UNSW-NB15 accuracy.
- The project improves the macro-F1-oriented position under a strict
  official-split protocol.
- Other very high UNSW-NB15 results are not used as direct numeric ranking
  targets when they use random splits, cross-validation, or binary settings.

## 8. Difficult-Class Behavior

Final W&B-tuned model, three-seed mean:

| Class | Precision | Recall | F1 | False-Positive Rate |
| --- | ---: | ---: | ---: | ---: |
| Analysis | 1.81% | 4.19% | 2.52% | 1.37% |
| Backdoor | 11.53% | 9.78% | 8.59% | 0.85% |
| DoS | 39.53% | 30.04% | 27.40% | 2.89% |
| Shellcode | 27.25% | 67.28% | 38.76% | 0.83% |
| Worms | 63.85% | 17.42% | 27.31% | 0.01% |

Main observation:

- Shellcode and Worms are handled better than Analysis and Backdoor.
- Analysis remains the major unresolved class.
- SMOTENC increased rare recall but caused precision collapse, so it was kept
  as an ablation rather than selected as the final model.

## 9. Operational Feasibility

| Model | Parameters | Model Size | Throughput | Interpretation |
| --- | ---: | ---: | ---: | --- |
| CNN | 42,250 | 0.16 MB | 40,460 samples/s | Fastest, weak macro-F1 |
| LSTM | 175,050 | 0.67 MB | 24,924 samples/s | Strong accuracy baseline |
| Feature Attention + CE | 311,370 | 1.19 MB | 9,749 samples/s | Strong pre-W&B model |
| Feature Attention + EQL v2 only | 311,370 | 1.19 MB | 10,640 samples/s | Strong imbalance ablation |
| **Final W&B Feature Attention + CE** | **1,453,290** | **5.54 MB** | **848 samples/s** | **More expensive but still practical for offline IDS benchmarking** |

The final model is heavier than the initial feature-attention model, but the
cost is reported transparently and remains feasible for thesis experimentation.

## 10. Contribution Summary

The UNSW-NB15 contribution is:

> A strict official-split multiclass IDS benchmark showing that a feature-axis
> multi-scale CNN-BiLSTM-attention architecture improves macro-F1 compared with
> internal deep-learning baselines and the closest official-split F1 reference,
> while exposing the remaining Analysis-class limitation and the precision cost
> of aggressive oversampling.

This supports the thesis because it shows:

- accuracy alone is not enough for multiclass IDS;
- macro-F1 and per-class metrics are necessary;
- loss-level and data-level imbalance handling must be compared carefully;
- aggressive oversampling can damage precision;
- operational metrics should be reported alongside detection metrics.

## 11. Limitations And Next Work

Limitations:

- Accuracy remains below Wiley FMB-IDM.
- Analysis remains poorly detected.
- The final W&B-tuned model is larger and slower than simpler baselines.
- Robustness to unseen attacks and feature perturbation is planned but not yet
  completed.

Next work:

- do not reopen UNSW model search unless specifically requested;
- use UNSW as the completed first dataset contribution;
- optionally add an unseen-attack holdout and perturbation test for Objective 3.

## References

- UNSW-NB15 dataset: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- Wiley FMB-IDM benchmark:
  <https://onlinelibrary.wiley.com/doi/10.1155/jece/2865894>
- Dual-Attention CNN-BiLSTM with EQL v2:
  <https://www.techscience.com/cmc/v86n1/64418/html>
- ADFCNN-BiLSTM with EQL v2:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11902464/>
