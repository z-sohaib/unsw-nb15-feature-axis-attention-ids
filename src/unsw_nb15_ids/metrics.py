from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )
    false_positive_rate = _false_positive_rates(matrix)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            class_name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "false_positive_rate": float(false_positive_rate[idx]),
                "support": int(support[idx]),
            }
            for idx, class_name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        ),
    }


def _false_positive_rates(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    false_positive_rates = []
    for idx in range(matrix.shape[0]):
        false_positive = matrix[:, idx].sum() - matrix[idx, idx]
        true_negative = total - matrix[idx, :].sum() - matrix[:, idx].sum() + matrix[idx, idx]
        denominator = false_positive + true_negative
        false_positive_rates.append(false_positive / denominator if denominator > 0 else 0.0)
    return np.asarray(false_positive_rates, dtype=np.float64)
