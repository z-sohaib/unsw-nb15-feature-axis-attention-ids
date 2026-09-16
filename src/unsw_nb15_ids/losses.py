from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class EQLv2Loss(nn.Module):
    """Practical EQL v2-style loss for long-tailed multiclass IDS experiments.

    The original EQL v2 was proposed for long-tailed object detection. This
    implementation keeps the core thesis idea: rare classes receive larger loss
    weights so majority classes do not dominate the gradient signal.
    """

    def __init__(self, labels: np.ndarray, num_classes: int, gamma: float = 2.0) -> None:
        super().__init__()
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        counts[counts == 0.0] = 1.0
        frequency = counts / counts.sum()
        inverse = (1.0 - frequency) ** gamma
        weights = inverse / inverse.mean()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights.to(logits.device))


class FocalLoss(nn.Module):
    def __init__(
        self,
        labels: np.ndarray,
        num_classes: int,
        gamma: float = 2.0,
        use_class_weights: bool = False,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        weights = class_weights(labels, num_classes) if use_class_weights else None
        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weights.to(logits.device) if self.weights is not None else None,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class ClassBalancedFocalLoss(nn.Module):
    """Focal loss with effective-number weights for long-tailed classes."""

    def __init__(
        self,
        labels: np.ndarray,
        num_classes: int,
        beta: float = 0.9999,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        counts[counts == 0.0] = 1.0
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.mean()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weights.to(logits.device),
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class BalancedSoftmaxLoss(nn.Module):
    """Balanced softmax loss for multiclass long-tailed recognition.

    The training logits are adjusted with the log class counts inside the loss.
    Evaluation still uses the raw model logits, which keeps the test protocol
    unchanged.
    """

    def __init__(self, labels: np.ndarray, num_classes: int) -> None:
        super().__init__()
        counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
        counts[counts == 0.0] = 1.0
        self.register_buffer("log_counts", torch.log(torch.tensor(counts, dtype=torch.float32)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_counts.to(logits.device)
        return F.cross_entropy(adjusted_logits, targets)


def build_loss(
    name: str,
    labels: np.ndarray,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights(labels, num_classes),
            label_smoothing=label_smoothing,
        )
    if name == "focal":
        return FocalLoss(labels, num_classes, use_class_weights=False)
    if name == "weighted_focal":
        return FocalLoss(labels, num_classes, use_class_weights=True)
    if name == "class_balanced_focal":
        return ClassBalancedFocalLoss(labels, num_classes)
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(labels, num_classes)
    if name == "eql_v2":
        return EQLv2Loss(labels, num_classes)
    raise ValueError(f"Unsupported loss: {name}")
