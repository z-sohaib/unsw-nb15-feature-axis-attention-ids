from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unsw_nb15_ids.config import load_config
from unsw_nb15_ids.data import load_unsw_nb15
from unsw_nb15_ids.metrics import classification_metrics
from unsw_nb15_ids.models import build_model
from unsw_nb15_ids.train import _loader, _predict
from unsw_nb15_ids.utils import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate UNSW-NB15 test-time perturbation robustness.")
    parser.add_argument("--config", required=True, help="Config used to train the selected model.")
    parser.add_argument("--model-path", required=True, help="Path to model.pt.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--noise-std", nargs="*", type=float, default=[0.01, 0.03, 0.05])
    parser.add_argument("--feature-dropout", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    print("[objective3] Loading and preprocessing UNSW-NB15 dataset", flush=True)
    bundle = load_unsw_nb15(config.data, seed=config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[objective3] Loaded test samples={len(bundle.y_test)} features={bundle.input_features} device={device}",
        flush=True,
    )
    model = build_model(config.model, input_features=bundle.input_features, num_classes=bundle.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("[objective3] Evaluating clean test set", flush=True)
    clean_metrics = _evaluate(model, bundle.x_test, bundle.y_test, bundle.class_names, config.training.batch_size, device)
    rng = np.random.default_rng(args.seed)
    results: dict[str, Any] = {
        "config": args.config,
        "model_path": args.model_path,
        "seed": args.seed,
        "clean": clean_metrics,
        "perturbations": [],
        "note": (
            "Perturbations are applied after preprocessing in model feature space. "
            "This is a controlled sensitivity test, not a fully realistic packet-level attack."
        ),
    }

    for std in args.noise_std:
        print(f"[objective3] Evaluating Gaussian noise std={std}", flush=True)
        perturbed = bundle.x_test + rng.normal(0.0, std, size=bundle.x_test.shape).astype(np.float32)
        metrics = _evaluate(model, perturbed.astype(np.float32), bundle.y_test, bundle.class_names, config.training.batch_size, device)
        results["perturbations"].append(_payload("gaussian_noise", std, clean_metrics, metrics))

    for rate in args.feature_dropout:
        print(f"[objective3] Evaluating feature dropout rate={rate}", flush=True)
        mask = rng.random(bundle.x_test.shape) >= rate
        perturbed = bundle.x_test * mask.astype(np.float32)
        metrics = _evaluate(model, perturbed.astype(np.float32), bundle.y_test, bundle.class_names, config.training.batch_size, device)
        results["perturbations"].append(_payload("feature_dropout", rate, clean_metrics, metrics))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved robustness evaluation to {output_path}")
    return 0


def _evaluate(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = _loader(x.astype(np.float32), y.astype(np.int64), batch_size, shuffle=False)
    y_true, y_pred = _predict(model, loader, device)
    return classification_metrics(y_true, y_pred, class_names)


def _payload(kind: str, value: float, clean: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": kind,
        "value": float(value),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy_drop": float(clean["accuracy"] - metrics["accuracy"]),
        "macro_f1_drop": float(clean["macro_f1"] - metrics["macro_f1"]),
        "weighted_f1_drop": float(clean["weighted_f1"] - metrics["weighted_f1"]),
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
