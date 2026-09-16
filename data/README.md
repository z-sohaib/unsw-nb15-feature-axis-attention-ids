# UNSW-NB15 Data

Place the official UNSW-NB15 CSV files here:

```text
data/raw/UNSW_NB15_training-set.csv
data/raw/UNSW_NB15_testing-set.csv
```

The loader expects the official columns, including:

- `label` for binary classification;
- `attack_cat` for multiclass classification;
- `proto`, `service`, and `state` as categorical flow features when present.

The `raw/` and `processed/` folders are ignored by Git.
