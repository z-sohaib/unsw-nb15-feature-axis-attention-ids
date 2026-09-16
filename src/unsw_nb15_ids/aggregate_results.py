from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate UNSW-NB15 experiment metrics.")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root directory containing experiment subdirectories.",
    )
    parser.add_argument(
        "--pattern",
        default="binary_*/metrics.json",
        help="Glob pattern relative to runs-dir.",
    )
    parser.add_argument(
        "--output",
        default="runs/binary_comparison.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke-test runs in the aggregate table.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = [
        row
        for path in sorted(runs_dir.glob(args.pattern))
        if (row := _row(path)) is not None
        and (args.include_smoke or not row["is_smoke_run"])
    ]
    if not rows:
        raise SystemExit(f"No metrics files matched {runs_dir / args.pattern}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model_name",
        "task",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "loss",
        "optimizer",
        "threshold_strategy",
        "calibrated_attack_threshold",
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
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")


def _row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    latency = metrics.get("latency", {})
    dataset = metrics.get("dataset", {})
    training = metrics.get("training_config", {})
    is_smoke_run = (
        "smoke" in str(metrics.get("experiment", "")).lower()
        or training.get("max_train_samples") is not None
    )
    return {
        "experiment": metrics.get("experiment", path.parent.name),
        "model_name": metrics.get("model_name", ""),
        "task": dataset.get("task", ""),
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "loss": training.get("loss", ""),
        "optimizer": training.get("optimizer", "adamw" if training else ""),
        "threshold_strategy": training.get("threshold_strategy", "argmax" if training else ""),
        "calibrated_attack_threshold": training.get("calibrated_attack_threshold", ""),
        "parameter_count": metrics.get("parameter_count", ""),
        "model_size_mb": metrics.get("model_size_mb", ""),
        "train_seconds": metrics.get("train_seconds", ""),
        "seconds_per_sample": latency.get("seconds_per_sample", ""),
        "samples_per_second": latency.get("samples_per_second", ""),
        "device": metrics.get("device", ""),
        "train_samples": dataset.get("train_samples", ""),
        "validation_samples": dataset.get("validation_samples", ""),
        "test_samples": dataset.get("test_samples", ""),
        "is_smoke_run": is_smoke_run,
    }


if __name__ == "__main__":
    main()
