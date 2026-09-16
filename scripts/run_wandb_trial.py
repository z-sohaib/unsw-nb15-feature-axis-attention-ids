from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unsw_nb15_ids.config import load_config
from unsw_nb15_ids.train import run as run_training


RARE_CLASSES = ("Analysis", "Backdoor", "Shellcode", "Worms")
DIFFICULT_CLASSES = ("Analysis", "Backdoor", "DoS", "Shellcode", "Worms")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one UNSW-NB15 W&B sweep trial.")
    parser.add_argument("--base-config", required=True, help="Base TOML config to override.")
    parser.add_argument("--project", default="pfe-thesis-unsw-nb15")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--run-prefix", default="sweep")
    args, unknown = parser.parse_known_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed. From projects/unsw_nb15, run "
            "`python -m pip install -e .` or `python -m pip install wandb`."
        ) from exc

    cli_params = _parse_unknown_args(unknown)
    with wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        config=cli_params,
        tags=["unsw-nb15", "multiclass", "hpo"],
    ) as wb_run:
        sweep_params = dict(wandb.config)
        resolved = _resolved_config(
            Path(args.base_config),
            sweep_params,
            run_id=wb_run.id,
            run_prefix=args.run_prefix,
        )
        resolved_dir = Path(resolved["experiment"]["output_dir"])
        resolved_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = resolved_dir / "resolved_config.toml"
        resolved_path.write_text(_toml_dumps(resolved), encoding="utf-8")
        wandb.config.update(_flatten_config(resolved), allow_val_change=True)

        config = load_config(resolved_path)
        run_training(config, config_path=resolved_path)

        metrics_path = resolved_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        flat_metrics = _flatten_metrics(metrics)
        wandb.log(flat_metrics)
        wandb.summary.update(flat_metrics)
        wandb.save(str(metrics_path))
        wandb.save(str(resolved_path))

    return 0


def _resolved_config(
    base_path: Path,
    params: dict[str, Any],
    run_id: str,
    run_prefix: str,
) -> dict[str, Any]:
    with base_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    for key, value in params.items():
        if key.startswith("_") or value is None:
            continue
        if key == "data.smote_profile":
            _apply_smote_profile(raw, str(value))
            continue
        _set_nested(raw, key, _normalize_value(key, value))

    base_name = str(raw["experiment"]["name"])
    raw["experiment"]["name"] = f"{run_prefix}_{base_name}_{run_id}"
    raw["experiment"]["output_dir"] = f"runs/wandb/{base_name}/{run_id}"
    return raw


def _apply_smote_profile(raw: dict[str, Any], profile: str) -> None:
    data = raw.setdefault("data", {})
    if profile == "none":
        data["apply_smote"] = False
        data["smote_method"] = "none"
        data.pop("smote_target_count", None)
        data.pop("smote_target_counts", None)
        return
    data["apply_smote"] = True
    data["smote_method"] = "smotenc"
    if profile == "global_3000":
        data["smote_target_count"] = 3000
        data.pop("smote_target_counts", None)
        return
    if profile == "global_6000":
        data["smote_target_count"] = 6000
        data.pop("smote_target_counts", None)
        return
    if profile == "perclass_conservative":
        data.pop("smote_target_count", None)
        data["smote_target_counts"] = {
            "Worms": 1200,
            "Shellcode": 6000,
            "Analysis": 6000,
            "Backdoor": 7000,
            "DoS": 9000,
        }
        return
    if profile == "perclass_moderate":
        data.pop("smote_target_count", None)
        data["smote_target_counts"] = {
            "Worms": 2000,
            "Shellcode": 5000,
            "Analysis": 5000,
            "Backdoor": 6000,
            "DoS": 8000,
        }
        return
    raise ValueError(f"Unsupported data.smote_profile={profile!r}.")


def _set_nested(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) < 2:
        return
    current = raw
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _normalize_value(key: str, value: Any) -> Any:
    if key == "model.kernel_sizes":
        return _as_list(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return _parse_jsonish(value)
    return value


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        parsed = _parse_jsonish(value)
        if isinstance(parsed, list):
            return parsed
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [value]


def _parse_unknown_args(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    index = 0
    while index < len(items):
        item = items[index]
        if not item.startswith("--"):
            index += 1
            continue
        payload = item[2:]
        if "=" in payload:
            key, value = payload.split("=", 1)
        else:
            key = payload
            if index + 1 < len(items) and not items[index + 1].startswith("--"):
                value = items[index + 1]
                index += 1
            else:
                value = "true"
        params[key] = _parse_jsonish(value)
        index += 1
    return params


def _parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(f"{prefix}.{child_key}" if prefix else child_key, child_value)
        else:
            flattened[prefix] = value

    visit("", raw)
    return flattened


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    per_class = metrics.get("per_class", {})
    rare = _class_means(per_class, RARE_CLASSES)
    difficult = _class_means(per_class, DIFFICULT_CLASSES)
    validation = _best_validation(metrics.get("history", []))
    flattened = {
        "test_accuracy": metrics.get("accuracy"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_weighted_f1": metrics.get("weighted_f1"),
        "rare_mean_precision": rare["precision"],
        "rare_mean_recall": rare["recall"],
        "rare_mean_f1": rare["f1"],
        "difficult_mean_precision": difficult["precision"],
        "difficult_mean_recall": difficult["recall"],
        "difficult_mean_f1": difficult["f1"],
        "best_val_macro_f1": validation.get("val_macro_f1"),
        "best_val_accuracy": validation.get("val_accuracy"),
        "best_val_weighted_f1": validation.get("val_weighted_f1"),
        "parameter_count": metrics.get("parameter_count"),
        "model_size_mb": metrics.get("model_size_mb"),
        "train_seconds": metrics.get("train_seconds"),
        "seconds_per_sample": metrics.get("latency", {}).get("seconds_per_sample"),
        "samples_per_second": metrics.get("latency", {}).get("samples_per_second"),
    }
    for class_name in ("Normal", "Generic", "Exploits", "Analysis", "Backdoor", "DoS", "Shellcode", "Worms"):
        values = _lookup_class(per_class, class_name)
        key = class_name.lower()
        flattened[f"{key}_precision"] = values.get("precision")
        flattened[f"{key}_recall"] = values.get("recall")
        flattened[f"{key}_f1"] = values.get("f1")
        flattened[f"{key}_fpr"] = values.get("false_positive_rate")
    return flattened


def _class_means(per_class: dict[str, Any], class_names: tuple[str, ...]) -> dict[str, float | None]:
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


def _mean(values: list[Any]) -> float | None:
    numeric = []
    for value in values:
        try:
            if value is not None:
                numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(numeric) / len(numeric) if numeric else None


def _best_validation(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    return max(history, key=lambda row: float(row.get("val_macro_f1", -1.0)))


def _toml_dumps(raw: dict[str, Any]) -> str:
    lines = []
    for section, values in raw.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        parts = [f"{key} = {_toml_value(item)}" for key, item in value.items()]
        return "{ " + ", ".join(parts) + " }"
    return json.dumps(str(value))


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - started
        print(f"Interrupted after {elapsed:.1f}s", file=sys.stderr)
        raise
