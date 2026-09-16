from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from unsw_nb15_ids.config import Config, load_config
from unsw_nb15_ids.data import load_unsw_nb15
from unsw_nb15_ids.losses import build_loss
from unsw_nb15_ids.metrics import classification_metrics
from unsw_nb15_ids.models import build_model
from unsw_nb15_ids.utils import ensure_dir, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train thesis IDS model.")
    parser.add_argument("--config", required=True, help="Path to TOML config.")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config, config_path=Path(args.config))


def run(config: Config, config_path: Path | None = None) -> None:
    set_seed(config.experiment.seed)
    output_dir = ensure_dir(config.experiment.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_unsw_nb15(config.data, seed=config.experiment.seed)
    bundle = _limit_bundle(bundle, config)
    model = build_model(
        config.model,
        input_features=bundle.input_features,
        num_classes=bundle.num_classes,
    ).to(device)

    loss_fn = build_loss(
        config.training.loss,
        bundle.y_train,
        bundle.num_classes,
        label_smoothing=config.training.label_smoothing,
    ).to(device)
    optimizer = _build_optimizer(config, model)

    train_loader = _loader(bundle.x_train, bundle.y_train, config.training.batch_size, shuffle=True)
    val_loader = _loader(bundle.x_val, bundle.y_val, config.training.batch_size, shuffle=False)
    test_loader = _loader(bundle.x_test, bundle.y_test, config.training.batch_size, shuffle=False)

    selection_metric = config.training.validation_selection_metric
    best_selection_score = -1.0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    train_start = time.perf_counter()
    print(
        f"Starting {config.experiment.name} on {device} | "
        f"train={len(bundle.y_train)} val={len(bundle.y_val)} test={len(bundle.y_test)} | "
        f"epochs={config.training.epochs} batch_size={config.training.batch_size}"
    )

    for epoch in range(1, config.training.epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=config.training.epochs,
        )
        y_val, pred_val = _predict(model, val_loader, device)
        val_metrics = classification_metrics(y_val, pred_val, bundle.class_names)
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_accuracy = float(val_metrics["accuracy"])
        val_weighted_f1 = float(val_metrics["weighted_f1"])
        selection_score = _validation_selection_score(val_metrics, selection_metric)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": val_weighted_f1,
                "val_accuracy": val_accuracy,
                "val_selection_score": selection_score,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} val_accuracy={val_accuracy:.4f} "
            f"selection_{selection_metric}={selection_score:.4f}"
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                print("Early stopping triggered.")
                break

    train_seconds = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    threshold = None
    multiclass_logit_bias = None
    calibration_metrics: dict[str, Any] = {}
    if bundle.num_classes == 2 and config.training.threshold_strategy != "argmax":
        y_val, _, val_probabilities = _predict(
            model,
            val_loader,
            device,
            return_probabilities=True,
        )
        threshold = _calibrate_binary_threshold(
            y_val,
            val_probabilities,
            bundle.class_names,
            config.training.threshold_strategy,
            config.training.threshold_min_normal_recall,
        )
    elif bundle.num_classes > 2 and config.training.multiclass_calibration_strategy != "none":
        y_val, val_logits = _collect_logits(model, val_loader, device)
        multiclass_logit_bias, calibration_metrics = _calibrate_multiclass_logit_bias(
            y_val,
            val_logits,
            bundle.class_names,
            config,
        )
    y_test, pred_test = _predict(
        model,
        test_loader,
        device,
        threshold=threshold,
        logit_bias=multiclass_logit_bias,
    )
    metrics = classification_metrics(y_test, pred_test, bundle.class_names)
    metrics["history"] = history
    metrics["experiment"] = config.experiment.name
    metrics["model_name"] = config.model.name
    metrics["device"] = str(device)
    metrics["device_info"] = _device_info(device)
    metrics["parameter_count"] = _parameter_count(model)
    metrics["model_size_mb"] = _model_size_mb(model)
    metrics["train_seconds"] = float(train_seconds)
    metrics["dataset"] = {
        "task": config.data.task,
        "class_names": bundle.class_names,
        "input_features": bundle.input_features,
        "train_samples": int(len(bundle.y_train)),
        "validation_samples": int(len(bundle.y_val)),
        "test_samples": int(len(bundle.y_test)),
    }
    metrics["preprocessing"] = {
        "split_indices_path": str(config.data.split_indices_path)
        if config.data.split_indices_path
        else None,
        "log_transform": config.data.log_transform,
        "preprocessor_path": str(config.data.preprocessor_path)
        if config.data.preprocessor_path
        else None,
        "apply_smote": config.data.apply_smote,
        "smote_method": config.data.smote_method,
        "smote_target_count": config.data.smote_target_count,
        "smote_target_counts": config.data.smote_target_counts,
    }
    metrics["training_config"] = {
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "loss": config.training.loss,
        "optimizer": config.training.optimizer,
        "validation_selection_metric": config.training.validation_selection_metric,
        "best_validation_selection_score": best_selection_score,
        "label_smoothing": config.training.label_smoothing,
        "early_stopping_patience": config.training.early_stopping_patience,
        "threshold_strategy": config.training.threshold_strategy,
        "threshold_min_normal_recall": config.training.threshold_min_normal_recall,
        "calibrated_attack_threshold": threshold,
        "multiclass_calibration_strategy": config.training.multiclass_calibration_strategy,
        "multiclass_calibration_classes": list(config.training.multiclass_calibration_classes),
        "multiclass_calibration_bias": None
        if multiclass_logit_bias is None
        else {
            class_name: float(multiclass_logit_bias[idx])
            for idx, class_name in enumerate(bundle.class_names)
            if float(multiclass_logit_bias[idx]) != 0.0
        },
        "multiclass_calibration_validation": calibration_metrics,
        "max_train_samples": config.training.max_train_samples,
        "max_val_samples": config.training.max_val_samples,
        "max_test_samples": config.training.max_test_samples,
    }
    metrics["latency"] = _measure_latency(model, test_loader, device)

    if config_path is not None:
        shutil.copy2(config_path, output_dir / "config.toml")
    write_json(output_dir / "metrics.json", metrics)
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _build_optimizer(config: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.training.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    if config.training.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")


def _validation_selection_score(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "macro_f1":
        return float(metrics["macro_f1"])
    if metric_name == "weighted_f1":
        return float(metrics["weighted_f1"])
    if metric_name == "accuracy":
        return float(metrics["accuracy"])
    raise ValueError(
        "Unsupported validation_selection_metric. "
        "Use one of: macro_f1, weighted_f1, accuracy."
    )


def _limit_bundle(bundle: Any, config: Config) -> Any:
    seed = config.experiment.seed
    bundle.x_train, bundle.y_train = _limit_split(
        bundle.x_train,
        bundle.y_train,
        config.training.max_train_samples,
        seed,
    )
    bundle.x_val, bundle.y_val = _limit_split(
        bundle.x_val,
        bundle.y_val,
        config.training.max_val_samples,
        seed + 1,
    )
    bundle.x_test, bundle.y_test = _limit_split(
        bundle.x_test,
        bundle.y_test,
        config.training.max_test_samples,
        seed + 2,
    )
    return bundle


def _limit_split(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    labels = np.unique(y)
    per_class_floor = max(1, max_samples // max(len(labels), 1))
    for label in labels:
        class_indices = np.flatnonzero(y == label)
        take = min(len(class_indices), per_class_floor)
        selected.append(rng.choice(class_indices, size=take, replace=False))
    selected_indices = np.concatenate(selected)
    remaining = max_samples - len(selected_indices)
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(y)), selected_indices, assume_unique=False)
        if len(available) > 0:
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            selected_indices = np.concatenate([selected_indices, extra])
    selected_indices = np.sort(selected_indices.astype(int))
    return x[selected_indices], y[selected_indices]


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    progress = tqdm(
        loader,
        desc=f"epoch {epoch:03d}/{total_epochs:03d}",
        leave=True,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    for x_batch, y_batch in progress:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(x_batch)
        total_items += len(x_batch)
        progress.set_postfix(loss=f"{total_loss / max(total_items, 1):.4f}")
    return total_loss / max(total_items, 1)


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None = None,
    logit_bias: np.ndarray | None = None,
    return_probabilities: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    bias_tensor = (
        torch.tensor(logit_bias, dtype=torch.float32, device=device)
        if logit_bias is not None
        else None
    )
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        if bias_tensor is not None:
            logits = logits + bias_tensor
        if threshold is not None and logits.shape[1] == 2:
            probability = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions = (probability >= threshold).astype(np.int64)
            probabilities.append(probability)
        else:
            predictions = logits.argmax(dim=1).cpu().numpy()
            if return_probabilities and logits.shape[1] == 2:
                probabilities.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
        y_pred.append(predictions)
        y_true.append(y_batch.numpy())
    y_true_array = np.concatenate(y_true)
    y_pred_array = np.concatenate(y_pred)
    if return_probabilities:
        return y_true_array, y_pred_array, np.concatenate(probabilities)
    return y_true_array, y_pred_array


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        batch_logits = model(x_batch.to(device)).cpu().numpy()
        logits.append(batch_logits)
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(logits)


def _calibrate_multiclass_logit_bias(
    y_true: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
    config: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    strategy = config.training.multiclass_calibration_strategy
    if strategy != "rare_logit_bias_grid":
        raise ValueError(f"Unsupported multiclass calibration strategy: {strategy}")
    if not config.training.multiclass_calibration_classes:
        raise ValueError("multiclass_calibration_classes must be set for rare_logit_bias_grid.")

    class_indices = []
    for class_name in config.training.multiclass_calibration_classes:
        if class_name not in class_names:
            raise ValueError(f"Unknown calibration class: {class_name}")
        class_indices.append(class_names.index(class_name))

    bias_values = np.arange(
        config.training.multiclass_calibration_bias_min,
        config.training.multiclass_calibration_bias_max
        + (config.training.multiclass_calibration_bias_step / 2.0),
        config.training.multiclass_calibration_bias_step,
    )
    best_bias = np.zeros(len(class_names), dtype=np.float32)
    best_metrics = classification_metrics(y_true, logits.argmax(axis=1), class_names)
    best_score = _multiclass_calibration_score(best_metrics, class_names)

    def search(position: int, current_bias: np.ndarray) -> None:
        nonlocal best_bias, best_metrics, best_score
        if position >= len(class_indices):
            predictions = (logits + current_bias).argmax(axis=1)
            metrics = classification_metrics(y_true, predictions, class_names)
            score = _multiclass_calibration_score(metrics, class_names)
            if score > best_score:
                best_score = score
                best_bias = current_bias.astype(np.float32).copy()
                best_metrics = metrics
            return
        class_idx = class_indices[position]
        for value in bias_values:
            next_bias = current_bias.copy()
            next_bias[class_idx] = float(value)
            search(position + 1, next_bias)

    search(0, np.zeros(len(class_names), dtype=np.float32))
    return best_bias, {
        "score": float(best_score),
        "macro_f1": float(best_metrics["macro_f1"]),
        "weighted_f1": float(best_metrics["weighted_f1"]),
        "per_class": best_metrics["per_class"],
    }


def _multiclass_calibration_score(metrics: dict[str, Any], class_names: list[str]) -> float:
    target_names = [name for name in ("Analysis", "Backdoor") if name in class_names]
    target_recall = float(
        np.mean([metrics["per_class"][name]["recall"] for name in target_names])
    ) if target_names else 0.0
    target_f1 = float(
        np.mean([metrics["per_class"][name]["f1"] for name in target_names])
    ) if target_names else 0.0
    macro_f1 = float(metrics["macro_f1"])
    return (0.55 * macro_f1) + (0.30 * target_f1) + (0.15 * target_recall)


def _calibrate_binary_threshold(
    y_true: np.ndarray,
    attack_probabilities: np.ndarray,
    class_names: list[str],
    strategy: str,
    min_normal_recall: float,
) -> float | None:
    if strategy == "argmax":
        return None
    if len(class_names) != 2:
        raise ValueError("Threshold calibration is only supported for binary tasks.")
    if strategy not in {"binary_macro_f1", "binary_macro_f1_min_normal_recall"}:
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    best_threshold = 0.5
    best_macro_f1 = -1.0
    fallback_threshold = 0.5
    fallback_macro_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (attack_probabilities >= threshold).astype(np.int64)
        metrics = classification_metrics(y_true, predictions, class_names)
        macro_f1 = float(metrics["macro_f1"])
        normal_recall = float(metrics["per_class"][class_names[0]]["recall"])
        if macro_f1 > fallback_macro_f1:
            fallback_macro_f1 = macro_f1
            fallback_threshold = float(threshold)
        if strategy == "binary_macro_f1_min_normal_recall" and normal_recall < min_normal_recall:
            continue
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)
    if best_macro_f1 < 0.0:
        return fallback_threshold
    return best_threshold


@torch.no_grad()
def _measure_latency(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, float]:
    model.eval()
    sample_count = 0
    start = time.perf_counter()
    for idx, (x_batch, _) in enumerate(loader):
        if idx >= max_batches:
            break
        _ = model(x_batch.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        sample_count += len(x_batch)
    elapsed = time.perf_counter() - start
    return {
        "measured_samples": float(sample_count),
        "total_seconds": float(elapsed),
        "seconds_per_sample": float(elapsed / max(sample_count, 1)),
        "samples_per_second": float(sample_count / elapsed) if elapsed > 0 else 0.0,
    }


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return float(total_bytes / (1024 * 1024))


def _device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_memory_total_mb": int(props.total_memory // (1024 * 1024)),
            }
        )
    return info


if __name__ == "__main__":
    main()
