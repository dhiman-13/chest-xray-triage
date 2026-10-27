"""Losses for multi-label classification with heavy class imbalance."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) applied independently to every label.

    (1 - p_t)^gamma down-weights easy examples, so the many obvious negatives stop
    dominating the gradient and the model focuses on hard / rare positives.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


def build_loss(name: str, pos_weight: list[float] | None, gamma: float = 2.0) -> nn.Module:
    pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight else None
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if name == "focal":
        return FocalLoss(gamma=gamma, pos_weight=pw)
    raise ValueError(f"Unknown loss '{name}' (expected 'bce' or 'focal')")
