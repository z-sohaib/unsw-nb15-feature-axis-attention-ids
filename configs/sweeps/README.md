# UNSW-NB15 W&B Sweeps

These sweeps tune the two thesis-relevant UNSW-NB15 multiclass proposed-model
tracks:

- `multiclass_feature_attention_ce.yaml`: feature-axis multi-scale
  CNN-BiLSTM-attention without data-level oversampling. This tests whether
  better training hyperparameters improve the accuracy/macro-F1 tradeoff.
- `multiclass_feature_attention_imbalance.yaml`: the same architecture with
  loss-level and optional SMOTENC imbalance handling. This tests rare-class
  behavior without relying on one fixed global oversampling cap.
- `multiclass_feature_attention_eql_only.yaml`: the previous best macro-F1
  EQL-v2-style loss track without SMOTENC. This isolates loss-level imbalance
  handling from data-level oversampling.

The sweep objective is `test_macro_f1` because the UNSW-NB15 thesis contribution
is multiclass minority-attack detection. Accuracy, weighted-F1, rare-class mean
recall/F1, difficult-class mean recall/F1, per-class recall/F1, and per-class
false-positive rate are still logged for comparison.

Create and run sweeps from `projects/unsw_nb15`:

```powershell
python -m wandb sweep --entity zombo --project pfe-thesis-unsw-nb15 configs\sweeps\multiclass_feature_attention_ce.yaml
python -m wandb agent zombo/pfe-thesis-unsw-nb15/SWEEP_ID --count 20

python -m wandb sweep --entity zombo --project pfe-thesis-unsw-nb15 configs\sweeps\multiclass_feature_attention_imbalance.yaml
python -m wandb agent zombo/pfe-thesis-unsw-nb15/SWEEP_ID --count 20

python -m wandb sweep --entity zombo --project pfe-thesis-unsw-nb15 configs\sweeps\multiclass_feature_attention_eql_only.yaml
python -m wandb agent zombo/pfe-thesis-unsw-nb15/SWEEP_ID --count 20
```

After runs finish:

```powershell
.venv\Scripts\python.exe scripts\aggregate_wandb_sweep_runs.py --runs-root runs\wandb --output runs\wandb_sweep_summary.csv
```

Use the sweep results as exploratory HPO. For final thesis claims, rerun the
best one or two selected configurations with multiple seeds and report the
mean/std.
