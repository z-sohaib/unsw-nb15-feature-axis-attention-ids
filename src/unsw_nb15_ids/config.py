from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    task: str
    validation_size: float
    apply_smote: bool
    smote_method: str = "none"
    smote_target_count: int | None = None
    smote_target_counts: dict[str, int] | None = None
    split_indices_path: Path | None = None
    log_transform: bool = False
    save_preprocessor: bool = False
    preprocessor_path: Path | None = None
    train_path: Path | None = None
    test_path: Path | None = None
    normal_path: Path | None = None
    attack_path: Path | None = None
    window_size: int = 100
    window_stride: int = 50
    label_column: str = "Normal/Attack"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    conv_channels: int
    kernel_sizes: tuple[int, ...]
    lstm_hidden: int
    lstm_layers: int
    attention_heads: int
    dense_hidden: int
    dropout: float
    transformer_dim: int | None = None
    transformer_layers: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    loss: str
    early_stopping_patience: int
    optimizer: str = "adamw"
    validation_selection_metric: str = "macro_f1"
    label_smoothing: float = 0.0
    threshold_strategy: str = "argmax"
    threshold_min_normal_recall: float = 0.0
    multiclass_calibration_strategy: str = "none"
    multiclass_calibration_classes: tuple[str, ...] = ()
    multiclass_calibration_bias_min: float = 0.0
    multiclass_calibration_bias_max: float = 4.0
    multiclass_calibration_bias_step: float = 0.25
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    experiment = ExperimentConfig(
        name=raw["experiment"]["name"],
        seed=int(raw["experiment"]["seed"]),
        output_dir=Path(raw["experiment"]["output_dir"]),
    )
    data_raw = raw["data"]
    data = DataConfig(
        dataset=data_raw["dataset"],
        task=data_raw["task"],
        validation_size=float(data_raw["validation_size"]),
        apply_smote=bool(data_raw.get("apply_smote", False)),
        smote_method=data_raw.get("smote_method", "none"),
        smote_target_count=_optional_int(data_raw.get("smote_target_count")),
        smote_target_counts=_optional_int_mapping(data_raw.get("smote_target_counts")),
        split_indices_path=_optional_path(data_raw.get("split_indices_path")),
        log_transform=bool(data_raw.get("log_transform", False)),
        save_preprocessor=bool(data_raw.get("save_preprocessor", False)),
        preprocessor_path=_optional_path(data_raw.get("preprocessor_path")),
        train_path=_optional_path(data_raw.get("train_path")),
        test_path=_optional_path(data_raw.get("test_path")),
        normal_path=_optional_path(data_raw.get("normal_path")),
        attack_path=_optional_path(data_raw.get("attack_path")),
        window_size=int(data_raw.get("window_size", 100)),
        window_stride=int(data_raw.get("window_stride", 50)),
        label_column=data_raw.get("label_column", "Normal/Attack"),
    )
    model_raw = raw["model"]
    model = ModelConfig(
        name=model_raw.get("name", "ms_cnn_bilstm_attention"),
        conv_channels=int(model_raw["conv_channels"]),
        kernel_sizes=tuple(int(k) for k in model_raw["kernel_sizes"]),
        lstm_hidden=int(model_raw["lstm_hidden"]),
        lstm_layers=int(model_raw["lstm_layers"]),
        attention_heads=int(model_raw["attention_heads"]),
        dense_hidden=int(model_raw["dense_hidden"]),
        dropout=float(model_raw["dropout"]),
        transformer_dim=_optional_int(model_raw.get("transformer_dim")),
        transformer_layers=int(model_raw.get("transformer_layers", 2)),
    )
    training_raw = raw["training"]
    training = TrainingConfig(
        epochs=int(training_raw["epochs"]),
        batch_size=int(training_raw["batch_size"]),
        learning_rate=float(training_raw["learning_rate"]),
        weight_decay=float(training_raw["weight_decay"]),
        loss=training_raw["loss"],
        early_stopping_patience=int(training_raw["early_stopping_patience"]),
        optimizer=training_raw.get("optimizer", "adamw"),
        validation_selection_metric=training_raw.get("validation_selection_metric", "macro_f1"),
        label_smoothing=float(training_raw.get("label_smoothing", 0.0)),
        threshold_strategy=training_raw.get("threshold_strategy", "argmax"),
        threshold_min_normal_recall=float(training_raw.get("threshold_min_normal_recall", 0.0)),
        multiclass_calibration_strategy=training_raw.get("multiclass_calibration_strategy", "none"),
        multiclass_calibration_classes=tuple(
            str(value) for value in training_raw.get("multiclass_calibration_classes", [])
        ),
        multiclass_calibration_bias_min=float(
            training_raw.get("multiclass_calibration_bias_min", 0.0)
        ),
        multiclass_calibration_bias_max=float(
            training_raw.get("multiclass_calibration_bias_max", 4.0)
        ),
        multiclass_calibration_bias_step=float(
            training_raw.get("multiclass_calibration_bias_step", 0.25)
        ),
        max_train_samples=_optional_int(training_raw.get("max_train_samples")),
        max_val_samples=_optional_int(training_raw.get("max_val_samples")),
        max_test_samples=_optional_int(training_raw.get("max_test_samples")),
    )
    return Config(experiment=experiment, data=data, model=model, training=training)


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_int_mapping(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Expected an inline TOML table for smote_target_counts.")
    return {str(key): int(item) for key, item in value.items()}
