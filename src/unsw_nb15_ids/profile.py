from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from unsw_nb15_ids.config import load_config
from unsw_nb15_ids.data import CATEGORICAL_COLUMNS, DROP_COLUMNS
from unsw_nb15_ids.utils import ensure_dir, require_file, write_json


@dataclass(frozen=True)
class SplitProfile:
    rows: int
    columns: int
    duplicate_rows: int
    total_missing_values: int
    columns_with_missing_values: dict[str, int]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile official UNSW-NB15 CSV files.")
    parser.add_argument(
        "--config",
        default="configs/multiclass.toml",
        help="Config containing train_path and test_path.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/profile",
        help="Directory for profile JSON, CSV, and figures.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)

    try:
        train_path = require_file(config.data.train_path, "UNSW-NB15 training CSV")
        test_path = require_file(config.data.test_path, "UNSW-NB15 testing CSV")
    except FileNotFoundError as exc:
        raise SystemExit(f"Dataset profile cannot run: {exc}") from exc

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    class_rows = _class_distribution_rows(train_df, test_df)
    _write_class_distribution(output_dir / "class_distribution.csv", class_rows)
    _plot_distribution(
        class_rows,
        task="binary",
        output_path=output_dir / "class_distribution_binary.png",
    )
    _plot_distribution(
        class_rows,
        task="multiclass",
        output_path=output_dir / "class_distribution_multiclass.png",
    )

    profile = {
        "source_files": {
            "train_path": str(train_path),
            "test_path": str(test_path),
        },
        "splits": {
            "train": asdict(_split_profile(train_df)),
            "test": asdict(_split_profile(test_df)),
        },
        "schema": _schema_profile(train_df, test_df),
        "class_distribution": _class_distribution_summary(class_rows),
        "phase_1_status": {
            "binary_labels_available": "label" in train_df.columns and "label" in test_df.columns,
            "multiclass_labels_available": (
                "attack_cat" in train_df.columns and "attack_cat" in test_df.columns
            ),
            "official_test_split_untouched": True,
        },
    }
    write_json(output_dir / "dataset_profile.json", profile)

    print(f"Wrote {output_dir / 'dataset_profile.json'}")
    print(f"Wrote {output_dir / 'class_distribution.csv'}")
    print(f"Wrote {output_dir / 'class_distribution_binary.png'}")
    print(f"Wrote {output_dir / 'class_distribution_multiclass.png'}")


def _split_profile(frame: pd.DataFrame) -> SplitProfile:
    missing = frame.isna().sum()
    missing_columns = {
        column: int(count)
        for column, count in missing.items()
        if int(count) > 0
    }
    return SplitProfile(
        rows=int(len(frame)),
        columns=int(len(frame.columns)),
        duplicate_rows=int(frame.duplicated().sum()),
        total_missing_values=int(missing.sum()),
        columns_with_missing_values=missing_columns,
    )


def _schema_profile(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    train_columns = list(train_df.columns)
    test_columns = list(test_df.columns)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in train_columns]
    feature_columns = [column for column in train_columns if column not in DROP_COLUMNS]
    numeric = [
        column
        for column in feature_columns
        if column not in categorical and pd.api.types.is_numeric_dtype(train_df[column])
    ]
    non_numeric_features = [
        column
        for column in feature_columns
        if column not in categorical and not pd.api.types.is_numeric_dtype(train_df[column])
    ]
    return {
        "train_columns": train_columns,
        "test_columns": test_columns,
        "same_columns_in_train_and_test": train_columns == test_columns,
        "feature_count_before_encoding": len(feature_columns),
        "feature_columns_before_encoding": feature_columns,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "unexpected_non_numeric_features": non_numeric_features,
        "label_columns": [column for column in ["label", "attack_cat"] if column in train_columns],
    }


def _class_distribution_rows(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for split_name, frame in [("train", train_df), ("test", test_df)]:
        if "label" in frame.columns:
            binary = frame["label"].astype(int).map({0: "normal", 1: "attack"})
            rows.extend(_counts_to_rows(binary, "binary", split_name))
        if "attack_cat" in frame.columns:
            multiclass = frame["attack_cat"].fillna("Normal").astype(str).str.strip()
            rows.extend(_counts_to_rows(multiclass, "multiclass", split_name))
    return rows


def _counts_to_rows(
    labels: pd.Series,
    task: str,
    split_name: str,
) -> list[dict[str, str | int | float]]:
    counts = labels.value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    return [
        {
            "task": task,
            "split": split_name,
            "class": str(label),
            "count": int(count),
            "percent": round(float(count) / max(total, 1) * 100.0, 6),
        }
        for label, count in counts.items()
    ]


def _write_class_distribution(
    path: Path,
    rows: list[dict[str, str | int | float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "split", "class", "count", "percent"])
        writer.writeheader()
        writer.writerows(rows)


def _class_distribution_summary(
    rows: list[dict[str, str | int | float]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        split_summary: dict[str, Any] = {}
        for split_name in sorted({str(row["split"]) for row in task_rows}):
            split_rows = [row for row in task_rows if row["split"] == split_name]
            counts = {str(row["class"]): int(row["count"]) for row in split_rows}
            non_zero_counts = [count for count in counts.values() if count > 0]
            split_summary[split_name] = {
                "class_count": len(counts),
                "counts": counts,
                "imbalance_ratio": (
                    round(max(non_zero_counts) / min(non_zero_counts), 6)
                    if non_zero_counts
                    else None
                ),
            }
        summary[task] = split_summary
    return summary


def _plot_distribution(
    rows: list[dict[str, str | int | float]],
    task: str,
    output_path: Path,
) -> None:
    task_rows = [row for row in rows if row["task"] == task]
    if not task_rows:
        return

    frame = pd.DataFrame(task_rows)
    pivot = frame.pivot(index="class", columns="split", values="count").fillna(0)
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)

    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.82)
    ax.set_title(f"UNSW-NB15 {task.capitalize()} Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Record Count")
    ax.tick_params(axis="x", labelrotation=45)
    ax.figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
