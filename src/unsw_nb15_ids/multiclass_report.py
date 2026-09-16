from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


RARE_CLASSES = ("Analysis", "Backdoor", "Shellcode", "Worms")
DIFFICULT_CLASSES = ("Analysis", "Backdoor", "DoS", "Shellcode", "Worms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UNSW-NB15 multiclass report artifacts.")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--pattern",
        default="multiclass_*/metrics.json",
        help="Glob pattern relative to runs-dir.",
    )
    parser.add_argument(
        "--output",
        default="runs/multiclass_comparison.csv",
        help="Wide CSV with one row per run.",
    )
    parser.add_argument(
        "--per-class-output",
        default="runs/multiclass_per_class.csv",
        help="Long CSV with one row per run and class.",
    )
    parser.add_argument(
        "--figures-dir",
        default="runs/multiclass_figures",
        help="Directory for confusion matrix figures.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke-test runs and limited-sample runs.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    metrics_paths = sorted(runs_dir.glob(args.pattern))
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for path in metrics_paths:
        metrics = _load_metrics(path)
        if not args.include_smoke and _is_smoke_run(metrics):
            continue
        if metrics.get("dataset", {}).get("task") != "multiclass":
            continue

        rows.append(_wide_row(path, metrics))
        per_class_rows.extend(_per_class_rows(metrics))
        _write_confusion_matrix(metrics, figures_dir)

    if not rows:
        raise SystemExit(f"No multiclass metrics files matched {runs_dir / args.pattern}")

    _write_csv(Path(args.output), rows, _wide_fieldnames(rows))
    _write_csv(
        Path(args.per_class_output),
        per_class_rows,
        [
            "experiment",
            "model_name",
            "loss",
            "smote_method",
            "smote_target_count",
            "smote_target_counts",
            "calibration_strategy",
            "class_name",
            "support",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
        ],
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.per_class_output}")
    print(f"Wrote confusion matrices to {figures_dir}")


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_smoke_run(metrics: dict[str, Any]) -> bool:
    training = metrics.get("training_config", {})
    return (
        "smoke" in str(metrics.get("experiment", "")).lower()
        or training.get("max_train_samples") is not None
    )


def _wide_row(path: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    dataset = metrics.get("dataset", {})
    training = metrics.get("training_config", {})
    preprocessing = metrics.get("preprocessing", {})
    latency = metrics.get("latency", {})
    per_class = _per_class_with_fpr(metrics)
    best_epoch, best_val_macro_f1 = _best_validation(metrics)
    rare_recall, rare_f1 = _class_group_means(metrics, RARE_CLASSES)
    difficult_recall, difficult_f1 = _class_group_means(metrics, DIFFICULT_CLASSES)
    row: dict[str, Any] = {
        "experiment": metrics.get("experiment", path.parent.name),
        "model_name": metrics.get("model_name", ""),
        "task": dataset.get("task", ""),
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "val_test_macro_f1_gap": _gap(best_val_macro_f1, metrics.get("macro_f1", "")),
        "rare_recall_mean": rare_recall,
        "rare_f1_mean": rare_f1,
        "difficult_recall_mean": difficult_recall,
        "difficult_f1_mean": difficult_f1,
        "loss": training.get("loss", ""),
        "optimizer": training.get("optimizer", ""),
        "validation_selection_metric": training.get("validation_selection_metric", "macro_f1"),
        "best_validation_selection_score": training.get("best_validation_selection_score", ""),
        "label_smoothing": training.get("label_smoothing", ""),
        "apply_smote": preprocessing.get("apply_smote", ""),
        "smote_method": preprocessing.get("smote_method", ""),
        "smote_target_count": preprocessing.get("smote_target_count", ""),
        "smote_target_counts": _json_string(preprocessing.get("smote_target_counts", "")),
        "calibration_strategy": training.get("multiclass_calibration_strategy", ""),
        "calibration_bias": _json_string(training.get("multiclass_calibration_bias", "")),
        "parameter_count": metrics.get("parameter_count", ""),
        "model_size_mb": metrics.get("model_size_mb", ""),
        "train_seconds": metrics.get("train_seconds", ""),
        "seconds_per_sample": latency.get("seconds_per_sample", ""),
        "samples_per_second": latency.get("samples_per_second", ""),
        "device": metrics.get("device", ""),
        "train_samples": dataset.get("train_samples", ""),
        "validation_samples": dataset.get("validation_samples", ""),
        "test_samples": dataset.get("test_samples", ""),
    }
    for class_name, values in per_class.items():
        key = _safe_name(class_name)
        row[f"precision_{key}"] = values.get("precision", "")
        row[f"recall_{key}"] = values.get("recall", "")
        row[f"f1_{key}"] = values.get("f1", "")
        row[f"fpr_{key}"] = values.get("false_positive_rate", "")
    return row


def _wide_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fixed = [
        "experiment",
        "model_name",
        "task",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "best_epoch",
        "best_val_macro_f1",
        "val_test_macro_f1_gap",
        "rare_recall_mean",
        "rare_f1_mean",
        "difficult_recall_mean",
        "difficult_f1_mean",
        "loss",
        "optimizer",
        "validation_selection_metric",
        "best_validation_selection_score",
        "label_smoothing",
        "apply_smote",
        "smote_method",
        "smote_target_count",
        "smote_target_counts",
        "calibration_strategy",
        "calibration_bias",
        "parameter_count",
        "model_size_mb",
        "train_seconds",
        "seconds_per_sample",
        "samples_per_second",
        "device",
        "train_samples",
        "validation_samples",
        "test_samples",
    ]
    dynamic = sorted({key for row in rows for key in row if key not in fixed})
    return fixed + dynamic


def _per_class_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    training = metrics.get("training_config", {})
    preprocessing = metrics.get("preprocessing", {})
    rows: list[dict[str, Any]] = []
    for class_name, values in _per_class_with_fpr(metrics).items():
        rows.append(
            {
                "experiment": metrics.get("experiment", ""),
                "model_name": metrics.get("model_name", ""),
                "loss": training.get("loss", ""),
                "smote_method": preprocessing.get("smote_method", ""),
                "smote_target_count": preprocessing.get("smote_target_count", ""),
                "smote_target_counts": _json_string(preprocessing.get("smote_target_counts", "")),
                "calibration_strategy": training.get("multiclass_calibration_strategy", ""),
                "class_name": class_name,
                "support": values.get("support", ""),
                "precision": values.get("precision", ""),
                "recall": values.get("recall", ""),
                "f1": values.get("f1", ""),
                "false_positive_rate": values.get("false_positive_rate", ""),
            }
        )
    return rows


def _per_class_with_fpr(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_class = {
        class_name: dict(values)
        for class_name, values in metrics.get("per_class", {}).items()
    }
    class_names = metrics.get("dataset", {}).get("class_names", [])
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=np.float64)
    if matrix.size == 0 or len(class_names) != matrix.shape[0]:
        return per_class
    total = matrix.sum()
    for idx, class_name in enumerate(class_names):
        false_positive = matrix[:, idx].sum() - matrix[idx, idx]
        true_negative = total - matrix[idx, :].sum() - matrix[:, idx].sum() + matrix[idx, idx]
        denominator = false_positive + true_negative
        fpr = false_positive / denominator if denominator > 0 else 0.0
        per_class.setdefault(class_name, {})["false_positive_rate"] = float(fpr)
    return per_class


def _best_validation(metrics: dict[str, Any]) -> tuple[float | str, float | str]:
    history = metrics.get("history", [])
    if not history:
        return "", ""
    best = max(history, key=lambda item: float(item.get("val_macro_f1", -1.0)))
    return best.get("epoch", ""), best.get("val_macro_f1", "")


def _class_group_means(
    metrics: dict[str, Any],
    class_names: tuple[str, ...],
) -> tuple[float | str, float | str]:
    per_class = metrics.get("per_class", {})
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_name in class_names:
        values = per_class.get(class_name)
        if values is None:
            continue
        recalls.append(float(values.get("recall", 0.0)))
        f1_scores.append(float(values.get("f1", 0.0)))
    if not recalls:
        return "", ""
    return float(np.mean(recalls)), float(np.mean(f1_scores))


def _gap(left: float | str, right: Any) -> float | str:
    if left == "" or right == "":
        return ""
    return float(left) - float(right)


def _write_confusion_matrix(metrics: dict[str, Any], output_dir: Path) -> None:
    class_names = metrics.get("dataset", {}).get("class_names", [])
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=np.float64)
    if matrix.size == 0 or not class_names:
        return

    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    experiment = _safe_name(str(metrics.get("experiment", "multiclass")))

    for values, suffix, fmt in [
        (matrix, "counts", ".0f"),
        (normalized, "normalized", ".2f"),
    ]:
        width = max(10, len(class_names) * 0.9)
        height = max(8, len(class_names) * 0.8)
        plt.figure(figsize=(width, height))
        sns.heatmap(
            values,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{metrics.get('experiment', 'multiclass')} confusion matrix")
        plt.tight_layout()
        plt.savefig(output_dir / f"{experiment}_confusion_matrix_{suffix}.png", dpi=180)
        plt.close()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _json_string(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    main()
