"""
Custom loss functions designed for high class-imbalance in medical multi-label classification.
"""

from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from medicalvision.config import DISEASE_CLASSES


def compute_class_weights(df: pd.DataFrame, disease_classes=DISEASE_CLASSES) -> torch.Tensor:
    """
    Compute positive class weight: w_pos = N_negative / N_positive.
    Capped at 50.0 to prevent gradient explosions on extremely rare classes.
    """
    weights = []
    total_samples = len(df)
    for disease in disease_classes:
        pos_count = df[disease].sum()
        if pos_count == 0:
            weights.append(1.0)
        else:
            neg_count = total_samples - pos_count
            w = float(neg_count) / float(pos_count)
            weights.append(min(w, 50.0))
    return torch.tensor(weights, dtype=torch.float32)


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy with Logits loss for multi-label classification.
    Applies pos_weight to each pathology class independently.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label Focal Loss to address extreme positive/negative imbalance.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = torch.pow((1.0 - p_t), self.gamma)

        loss = alpha_factor * modulating_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
