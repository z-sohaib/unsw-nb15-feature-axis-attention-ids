from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unsw_nb15_ids.config import Config, load_config
from unsw_nb15_ids.data import (
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    _apply_smotenc,
    _apply_smote,
    _build_preprocessor,
    _labels,
    _log1p_nonnegative,
    _save_preprocessor,
    _split_indices,
    _to_dense_float32,
)
from unsw_nb15_ids.utils import ensure_dir, require_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate UNSW-NB15 preprocessing before training."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/binary.toml", "configs/multiclass.toml"],
        help="One or more experiment configs to validate.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/preprocessing",
        help="Directory for preprocessing validation artifacts.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    configs = [load_config(path) for path in args.configs]
    results = [_validate_config(config) for config in configs]

    write_json(
        output_dir / "preprocessing_summary.json",
        {
            "phase": "phase_2_preprocessing_validation",
            "status": "completed",
            "validated_configs": [config.experiment.name for config in configs],
            "results": results,
        },
    )
    _write_distribution_table(output_dir / "split_class_distribution.csv", results)
    _write_feature_table(output_dir / "encoded_features.csv", results)

    print(f"Wrote {output_dir / 'preprocessing_summary.json'}")
    print(f"Wrote {output_dir / 'split_class_distribution.csv'}")
    print(f"Wrote {output_dir / 'encoded_features.csv'}")


def _validate_config(config: Config) -> dict[str, Any]:
    train_path = require_file(config.data.train_path, "UNSW-NB15 training CSV")
    test_path = require_file(config.data.test_path, "UNSW-NB15 testing CSV")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train_full, class_names = _labels(train_df, config.data.task)
    y_test, _ = _labels(test_df, config.data.task, class_names=class_names)

    x_train_full = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
    x_test_df = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])

    train_indices, val_indices = _split_indices(train_df, config.data, config.experiment.seed)
    x_train_df = x_train_full.iloc[train_indices].copy()
    x_val_df = x_train_full.iloc[val_indices].copy()
    y_train = y_train_full[train_indices]
    y_val = y_train_full[val_indices]

    preprocessor = _build_preprocessor(x_train_df, log_transform=config.data.log_transform)
    preprocessor.fit(x_train_df)
    if config.data.save_preprocessor:
        _save_preprocessor(preprocessor, config.data, x_train_df)

    x_train_encoded = _to_dense_float32(preprocessor.transform(x_train_df))
    x_val_encoded = _to_dense_float32(preprocessor.transform(x_val_df))
    x_test_encoded = _to_dense_float32(preprocessor.transform(x_test_df))

    smote_summary = _smote_summary(
        config=config,
        x_train_df=x_train_df,
        preprocessor=preprocessor,
        x_train_encoded=x_train_encoded,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    categorical_summary = _categorical_summary(preprocessor, x_train_df, x_val_df, x_test_df)
    scaling_summary = _scaling_summary(preprocessor, x_train_df)

    return {
        "experiment": config.experiment.name,
        "task": config.data.task,
        "seed": config.experiment.seed,
        "source_files": {
            "train_path": str(train_path),
            "test_path": str(test_path),
        },
        "split_policy": {
            "validation_size": config.data.validation_size,
            "validation_created_from_official_train_only": True,
            "official_test_used_for_validation": False,
            "train_validation_indices_disjoint": len(set(x_train_df.index) & set(x_val_df.index)) == 0,
            "train_plus_validation_equals_official_train": (
                len(x_train_df) + len(x_val_df) == len(train_df)
            ),
            "official_test_rows": len(test_df),
        },
        "class_names": class_names,
        "class_distribution": {
            "train": _distribution(y_train, class_names),
            "validation": _distribution(y_val, class_names),
            "test": _distribution(y_test, class_names),
        },
        "preprocessing": {
            "feature_count_before_encoding": int(x_train_full.shape[1]),
            "log_transform": bool(config.data.log_transform),
            "encoded_feature_count": int(x_train_encoded.shape[1]),
            "train_shape_after_encoding": list(x_train_encoded.shape),
            "validation_shape_after_encoding": list(x_val_encoded.shape),
            "test_shape_after_encoding": list(x_test_encoded.shape),
            "dtype_after_encoding": str(x_train_encoded.dtype),
            "categorical": categorical_summary,
            "numeric_scaling": scaling_summary,
            "preprocessor_saved": bool(config.data.save_preprocessor),
            "preprocessor_path": str(config.data.preprocessor_path)
            if config.data.preprocessor_path
            else None,
        },
        "balancing": smote_summary,
    }


def _categorical_summary(
    preprocessor: Any,
    x_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
) -> dict[str, Any]:
    categorical_columns = [c for c in CATEGORICAL_COLUMNS if c in x_train_df.columns]
    encoder = preprocessor.named_transformers_["categorical"]
    categories_by_column: dict[str, Any] = {}

    for column, categories in zip(categorical_columns, encoder.categories_, strict=True):
        train_values = set(x_train_df[column].astype(str).unique())
        val_values = set(x_val_df[column].astype(str).unique())
        test_values = set(x_test_df[column].astype(str).unique())
        encoded_categories = [str(value) for value in categories.tolist()]
        categories_by_column[column] = {
            "encoded_category_count": len(encoded_categories),
            "encoded_categories": encoded_categories,
            "categories_fit_from_train_only": set(encoded_categories) == train_values,
            "validation_unseen_category_count": len(val_values - train_values),
            "test_unseen_category_count": len(test_values - train_values),
            "handle_unknown": encoder.handle_unknown,
        }

    return {
        "categorical_columns": categorical_columns,
        "categorical_feature_count_after_one_hot": int(
            sum(len(categories) for categories in encoder.categories_)
        ),
        "columns": categories_by_column,
    }


def _scaling_summary(preprocessor: Any, x_train_df: pd.DataFrame) -> dict[str, Any]:
    categorical_columns = [c for c in CATEGORICAL_COLUMNS if c in x_train_df.columns]
    numeric_columns = [c for c in x_train_df.columns if c not in categorical_columns]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scaler"]
    train_numeric = x_train_df[numeric_columns].astype(float)
    log_transform = "log1p" in numeric_pipeline.named_steps
    if log_transform:
        train_numeric_values = _log1p_nonnegative(train_numeric)
    else:
        train_numeric_values = train_numeric.to_numpy(dtype=float)

    train_means = train_numeric_values.mean(axis=0)
    train_scales = train_numeric_values.std(axis=0, ddof=0)
    train_scales = np.where(train_scales == 0.0, 1.0, train_scales)
    mean_diff = np.abs(scaler.mean_ - train_means)
    scale_diff = np.abs(scaler.scale_ - train_scales)

    return {
        "numeric_columns": numeric_columns,
        "numeric_feature_count": len(numeric_columns),
        "log1p_before_scaling": log_transform,
        "scaler": "StandardScaler",
        "scaler_fit_on_training_split_only": bool(
            np.allclose(scaler.mean_, train_means) and np.allclose(scaler.scale_, train_scales)
        ),
        "max_abs_train_mean_difference": float(mean_diff.max(initial=0.0)),
        "max_abs_train_scale_difference": float(scale_diff.max(initial=0.0)),
    }


def _smote_summary(
    config: Config,
    x_train_df: pd.DataFrame,
    preprocessor: Any,
    x_train_encoded: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    before = {
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
    }
    summary: dict[str, Any] = {
        "configured_apply_smote": bool(config.data.apply_smote),
        "smote_method": config.data.smote_method,
        "applies_to_training_split_only": True,
        "validation_rows_unchanged_by_balancing": True,
        "test_rows_unchanged_by_balancing": True,
        "before_balancing": before,
    }

    if not config.data.apply_smote:
        summary["after_balancing"] = before
        return summary

    if config.data.smote_method == "smotenc":
        x_balanced_df, y_balanced = _apply_smotenc(
            x_train_df,
            y_train,
            seed=config.experiment.seed,
        )
        x_balanced = preprocessor.transform(x_balanced_df)
    elif config.data.smote_method == "encoded_smote":
        x_balanced, y_balanced = _apply_smote(
            x_train_encoded,
            y_train,
            seed=config.experiment.seed,
        )
    else:
        raise ValueError(f"Unsupported SMOTE method: {config.data.smote_method}")
    summary["after_balancing"] = {
        "train_rows": int(len(y_balanced)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "encoded_feature_count": int(x_balanced.shape[1]),
    }
    summary["training_rows_changed_by_smote"] = int(len(y_balanced)) != int(len(y_train))
    return summary


def _distribution(y: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    counts = np.bincount(y.astype(int), minlength=len(class_names))
    total = int(counts.sum())
    non_zero = [int(count) for count in counts if count > 0]
    return {
        "total": total,
        "counts": {
            class_name: int(count)
            for class_name, count in zip(class_names, counts.tolist(), strict=True)
        },
        "percent": {
            class_name: round(float(count) / max(total, 1) * 100.0, 6)
            for class_name, count in zip(class_names, counts.tolist(), strict=True)
        },
        "imbalance_ratio": (
            round(max(non_zero) / min(non_zero), 6)
            if non_zero
            else None
        ),
    }


def _write_distribution_table(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        for split, distribution in result["class_distribution"].items():
            for class_name, count in distribution["counts"].items():
                rows.append(
                    {
                        "task": result["task"],
                        "experiment": result["experiment"],
                        "split": split,
                        "class": class_name,
                        "count": count,
                        "percent": distribution["percent"][class_name],
                    }
                )
    _write_csv(path, rows, ["task", "experiment", "split", "class", "count", "percent"])


def _write_feature_table(path: Path, results: list[dict[str, Any]]) -> None:
    rows = [
        {
            "task": result["task"],
            "experiment": result["experiment"],
            "feature_count_before_encoding": result["preprocessing"]["feature_count_before_encoding"],
            "encoded_feature_count": result["preprocessing"]["encoded_feature_count"],
            "categorical_feature_count_after_one_hot": result["preprocessing"]["categorical"][
                "categorical_feature_count_after_one_hot"
            ],
            "numeric_feature_count": result["preprocessing"]["numeric_scaling"][
                "numeric_feature_count"
            ],
        }
        for result in results
    ]
    _write_csv(
        path,
        rows,
        [
            "task",
            "experiment",
            "feature_count_before_encoding",
            "encoded_feature_count",
            "categorical_feature_count_after_one_hot",
            "numeric_feature_count",
        ],
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
