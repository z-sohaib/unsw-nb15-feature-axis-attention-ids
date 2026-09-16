from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unsw_nb15_ids.bundle import DatasetBundle
from unsw_nb15_ids.config import load_config
from unsw_nb15_ids.data import DROP_COLUMNS, _as_sequence, _build_preprocessor, _labels, _to_dense_float32
from unsw_nb15_ids.losses import build_loss
from unsw_nb15_ids.metrics import classification_metrics
from unsw_nb15_ids.models import build_model
from unsw_nb15_ids.train import (
    _build_optimizer,
    _device_info,
    _loader,
    _measure_latency,
    _model_size_mb,
    _parameter_count,
    _predict,
    _train_one_epoch,
    _validation_selection_score,
)
from unsw_nb15_ids.utils import ensure_dir, require_file, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Train UNSW-NB15 with one attack category held out.")
    parser.add_argument("--config", required=True, help="Base final model config.")
    parser.add_argument("--heldout-category", required=True, help="Attack category removed from training.")
    parser.add_argument("--output-dir", required=True, help="Output run directory.")
    parser.add_argument("--epochs", type=int, default=25, help="Bounded diagnostic epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Bounded diagnostic batch size.")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-val-samples", type=int, default=12000)
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Disable cuDNN kernels for recurrent layers if the GPU backend is unstable.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. Use cpu if the Windows CUDA backend is unstable for full holdout runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/checkpoint.pt if it exists.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_holdout(
        config,
        Path(args.config),
        args.heldout_category,
        Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        disable_cudnn=args.disable_cudnn,
        requested_device=args.device,
        resume=args.resume,
    )
    return 0


def run_holdout(
    config: Any,
    config_path: Path,
    heldout_category: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    disable_cudnn: bool = False,
    requested_device: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    set_seed(config.experiment.seed)
    if disable_cudnn and torch.cuda.is_available():
        # The full unseen-holdout run is long and recurrent-layer heavy. On
        # Windows, cuDNN can raise intermittent internal errors during LSTM
        # backpropagation, so this diagnostic path uses the stable PyTorch
        # kernels while keeping CUDA acceleration.
        torch.backends.cudnn.enabled = False
    output_dir = ensure_dir(output_dir)
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle, protocol_summary = _load_holdout_bundle(config, heldout_category)
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

    bundle.x_train, bundle.y_train = _limit_split(
        bundle.x_train,
        bundle.y_train,
        max_train_samples,
        config.experiment.seed,
    )
    bundle.x_val, bundle.y_val = _limit_split(
        bundle.x_val,
        bundle.y_val,
        max_val_samples,
        config.experiment.seed + 1,
    )

    train_loader = _loader(bundle.x_train, bundle.y_train, batch_size, shuffle=True)
    val_loader = _loader(bundle.x_val, bundle.y_val, batch_size, shuffle=False)
    test_loader = _loader(bundle.x_test, bundle.y_test, batch_size, shuffle=False)

    best_selection_score = -1.0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    start_epoch = 1
    start = time.perf_counter()
    selection_metric = config.training.validation_selection_metric
    checkpoint_path = output_dir / "checkpoint.pt"
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_state = checkpoint.get("best_state")
        best_selection_score = float(checkpoint.get("best_selection_score", -1.0))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("next_epoch", 1))
        print(
            f"Resuming from {checkpoint_path} at epoch {start_epoch} "
            f"with best {selection_metric}={best_selection_score:.4f}"
        )

    print(
        f"Starting UNSW-NB15 unseen holdout={heldout_category} on {device} | "
        f"train={len(bundle.y_train)} val={len(bundle.y_val)} test={len(bundle.y_test)} | "
        f"epochs={epochs} batch_size={batch_size}"
    )
    for epoch in range(start_epoch, epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=epochs,
        )
        y_val, pred_val = _predict(model, val_loader, device)
        val_metrics = classification_metrics(y_val, pred_val, bundle.class_names)
        selection_score = _validation_selection_score(val_metrics, selection_metric)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_weighted_f1": float(val_metrics["weighted_f1"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_selection_score": float(selection_score),
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"selection_{selection_metric}={selection_score:.4f}"
        )
        if selection_score > best_selection_score:
            best_selection_score = float(selection_score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_model_partial.pt")
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                print("Early stopping triggered.")
                break
        (output_dir / "partial_history.json").write_text(
            json.dumps(
                {
                    "heldout_category": heldout_category,
                    "completed_epochs": epoch,
                    "best_validation_selection_score": best_selection_score,
                    "selection_metric": selection_metric,
                    "history": history,
                    "training_config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "max_train_samples": max_train_samples,
                        "max_val_samples": max_val_samples,
                        "requested_device": requested_device,
                        "disable_cudnn": disable_cudnn,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        torch.save(
            {
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_selection_score": best_selection_score,
                "stale_epochs": stale_epochs,
                "history": history,
                "next_epoch": epoch + 1,
                "heldout_category": heldout_category,
                "selection_metric": selection_metric,
            },
            checkpoint_path,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, pred_test = _predict(model, test_loader, device)
    metrics = classification_metrics(y_test, pred_test, bundle.class_names)
    metrics.update(
        {
            "experiment": f"{config.experiment.name}_unseen_{heldout_category}",
            "model_name": config.model.name,
            "device": str(device),
            "device_info": _device_info(device),
            "parameter_count": _parameter_count(model),
            "model_size_mb": _model_size_mb(model),
            "train_seconds": float(time.perf_counter() - start),
            "history": history,
            "training_config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
                "loss": config.training.loss,
                "optimizer": config.training.optimizer,
                "validation_selection_metric": selection_metric,
                "best_validation_selection_score": best_selection_score,
                "label_smoothing": config.training.label_smoothing,
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
            },
            "objective3_protocol": protocol_summary,
            "heldout_analysis": _heldout_analysis(
                y_true=y_test,
                y_pred=pred_test,
                class_names=bundle.class_names,
                heldout_category=heldout_category,
            ),
            "latency": _measure_latency(model, test_loader, device),
        }
    )

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), output_dir / "model.pt")
    shutil.copy2(config_path, output_dir / "base_config.toml")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    return metrics


def _limit_split(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(y):
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


def _load_holdout_bundle(config: Any, heldout_category: str) -> tuple[DatasetBundle, dict[str, Any]]:
    train_path = require_file(config.data.train_path, "UNSW-NB15 training CSV")
    test_path = require_file(config.data.test_path, "UNSW-NB15 testing CSV")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    full_train_labels = train_df["attack_cat"].fillna("Normal").astype(str).str.strip()
    test_labels = test_df["attack_cat"].fillna("Normal").astype(str).str.strip()
    class_names = sorted((set(full_train_labels) | set(test_labels)))
    if heldout_category not in class_names:
        raise ValueError(f"Held-out category {heldout_category!r} not found. Available: {class_names}")

    known_train_df = train_df.loc[full_train_labels != heldout_category].copy()
    y_known, _ = _labels(known_train_df, "multiclass", class_names=class_names)
    x_known_df = known_train_df.drop(columns=[column for column in DROP_COLUMNS if column in known_train_df.columns])
    x_test_df = test_df.drop(columns=[column for column in DROP_COLUMNS if column in test_df.columns])
    y_test, _ = _labels(test_df, "multiclass", class_names=class_names)

    indices = np.arange(len(known_train_df))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config.data.validation_size,
        random_state=config.experiment.seed,
        stratify=y_known,
    )
    train_indices = np.sort(train_indices.astype(int))
    val_indices = np.sort(val_indices.astype(int))
    x_train_df = x_known_df.iloc[train_indices].copy()
    x_val_df = x_known_df.iloc[val_indices].copy()
    y_train = y_known[train_indices]
    y_val = y_known[val_indices]

    preprocessor = _build_preprocessor(x_train_df, log_transform=config.data.log_transform)
    x_train = _to_dense_float32(preprocessor.fit_transform(x_train_df))
    x_val = _to_dense_float32(preprocessor.transform(x_val_df))
    x_test = _to_dense_float32(preprocessor.transform(x_test_df))

    summary = {
        "mode": "unsw_nb15_unseen_attack_holdout",
        "heldout_category": heldout_category,
        "class_names": class_names,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "known_train_rows_before_validation_split": int(len(known_train_df)),
        "heldout_training_rows_removed": int((full_train_labels == heldout_category).sum()),
        "heldout_test_rows": int((test_labels == heldout_category).sum()),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "protocol": [
            "The official training CSV is loaded first.",
            "The selected attack category is removed from training and validation.",
            "The official test CSV remains untouched and still contains the held-out category.",
            "Preprocessing is fitted only on known-category training rows.",
            "Held-out recall is reported both as exact-class recall and as suspicious non-Normal recall.",
        ],
    }
    return (
        DatasetBundle(
            x_train=_as_sequence(x_train),
            y_train=y_train.astype(np.int64),
            x_val=_as_sequence(x_val),
            y_val=y_val.astype(np.int64),
            x_test=_as_sequence(x_test),
            y_test=y_test.astype(np.int64),
            class_names=class_names,
        ),
        summary,
    )


def _heldout_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    heldout_category: str,
) -> dict[str, Any]:
    heldout_idx = class_names.index(heldout_category)
    normal_idx = class_names.index("Normal") if "Normal" in class_names else None
    mask = y_true == heldout_idx
    if not np.any(mask):
        return {
            "heldout_category": heldout_category,
            "heldout_support": 0,
            "exact_recall": None,
            "suspicious_non_normal_recall": None,
            "normal_confusion_rate": None,
        }
    heldout_pred = y_pred[mask]
    exact_recall = float(np.mean(heldout_pred == heldout_idx))
    suspicious_recall = None if normal_idx is None else float(np.mean(heldout_pred != normal_idx))
    normal_confusion = None if normal_idx is None else float(np.mean(heldout_pred == normal_idx))
    return {
        "heldout_category": heldout_category,
        "heldout_support": int(mask.sum()),
        "exact_recall": exact_recall,
        "suspicious_non_normal_recall": suspicious_recall,
        "normal_confusion_rate": normal_confusion,
        "prediction_distribution_on_heldout": {
            class_names[index]: int(count)
            for index, count in zip(*np.unique(heldout_pred, return_counts=True))
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
