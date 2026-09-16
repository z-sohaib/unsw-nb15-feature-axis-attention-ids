from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RARE_CLASSES = ("Analysis", "Backdoor", "Shellcode", "Worms")
DIFFICULT_CLASSES = ("Analysis", "Backdoor", "DoS", "Shellcode", "Worms")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate local UNSW-NB15 W&B sweep metrics.")
    parser.add_argument("--runs-root", default="runs/wandb")
    parser.add_argument("--output", default="runs/wandb_sweep_summary.csv")
    parser.add_argument("--sort-by", default="test_macro_f1")
    args = parser.parse_args()

    rows = [_row(path) for path in Path(args.runs_root).glob("*/*/metrics.json")]
    rows.sort(key=lambda row: _float(row.get(args.sort_by)), reverse=True)
    _write_csv(Path(args.output), rows)
    print(f"Wrote {args.output} with {len(rows)} rows")
    return 0


def _row(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_class = metrics.get("per_class", {})
    rare = _class_means(per_class, RARE_CLASSES)
    difficult = _class_means(per_class, DIFFICULT_CLASSES)
    training = metrics.get("training_config", {})
    latency = metrics.get("latency", {})
    config_path = metrics_path.with_name("config.toml")
    if not config_path.exists():
        config_path = metrics_path.with_name("resolved_config.toml")
    row: dict[str, Any] = {
        "experiment": metrics.get("experiment", metrics_path.parent.name),
        "model_name": metrics.get("model_name", ""),
        "task": metrics.get("dataset", {}).get("task", ""),
        "loss": training.get("loss", ""),
        "optimizer": training.get("optimizer", ""),
        "validation_selection_metric": training.get("validation_selection_metric", ""),
        "label_smoothing": training.get("label_smoothing", ""),
        "batch_size": training.get("batch_size", ""),
        "learning_rate": training.get("learning_rate", ""),
        "weight_decay": training.get("weight_decay", ""),
        "test_accuracy": metrics.get("accuracy", ""),
        "test_macro_f1": metrics.get("macro_f1", ""),
        "test_weighted_f1": metrics.get("weighted_f1", ""),
        "rare_mean_precision": rare["precision"],
        "rare_mean_recall": rare["recall"],
        "rare_mean_f1": rare["f1"],
        "difficult_mean_precision": difficult["precision"],
        "difficult_mean_recall": difficult["recall"],
        "difficult_mean_f1": difficult["f1"],
        "parameter_count": metrics.get("parameter_count", ""),
        "model_size_mb": metrics.get("model_size_mb", ""),
        "train_seconds": metrics.get("train_seconds", ""),
        "seconds_per_sample": latency.get("seconds_per_sample", ""),
        "samples_per_second": latency.get("samples_per_second", ""),
        "metrics_path": str(metrics_path),
        "config_path": str(config_path),
    }
    for class_name in ("Normal", "Generic", "Exploits", "Analysis", "Backdoor", "DoS", "Shellcode", "Worms"):
        values = _lookup_class(per_class, class_name)
        key = class_name.lower()
        row[f"{key}_precision"] = values.get("precision", "")
        row[f"{key}_recall"] = values.get("recall", "")
        row[f"{key}_f1"] = values.get("f1", "")
        row[f"{key}_fpr"] = values.get("false_positive_rate", "")
    return row


def _class_means(per_class: dict[str, Any], class_names: tuple[str, ...]) -> dict[str, float | str]:
    rows = [_lookup_class(per_class, name) for name in class_names]
    return {
        metric: _mean([row.get(metric) for row in rows])
        for metric in ("precision", "recall", "f1")
    }


def _lookup_class(per_class: dict[str, Any], class_name: str) -> dict[str, Any]:
    if class_name in per_class:
        return per_class[class_name]
    lower_map = {str(key).lower(): value for key, value in per_class.items()}
    return lower_map.get(class_name.lower(), {})


def _mean(values: list[Any]) -> float | str:
    numeric = []
    for value in values:
        try:
            if value not in {"", None}:
                numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(numeric) / len(numeric) if numeric else ""


def _float(value: Any) -> float:
    try:
        if value in {"", None}:
            return -1.0
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
