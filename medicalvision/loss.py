"""
MedicalVision AI - Disentangled Anatomical Contrastive Loss.
Implements pairwise margin regularization to decouple mutually confusable or
co-occurring pathologies (specifically Cardiomegaly and Pleural Effusion).
"""

import torch
import torch.nn as nn
from typing import Optional


class DisentangledBCEWithLogitsLoss(nn.Module):
    """
    Multi-label loss combining weighted BCE with pairwise contrastive margin penalty.
    When a scan is positive for condition A (e.g. Cardiomegaly) and negative for
    condition B (e.g. Effusion), any predicted probability for condition B above
    the margin delta is quadratically penalized.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        cardio_idx: int = 4,
        eff_idx: int = 3,
        margin: float = 0.08,
        lambda_contrast: float = 2.0,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cardio_idx = cardio_idx
        self.eff_idx = eff_idx
        self.margin = margin
        self.lambda_contrast = lambda_contrast

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        y_c = targets[:, self.cardio_idx]
        y_e = targets[:, self.eff_idx]
        p_c = probs[:, self.cardio_idx]
        p_e = probs[:, self.eff_idx]

        # Case 1: Pure Cardiomegaly (y_c=1, y_e=0) -> Suppress Effusion probability
        mask_c_only = (y_c == 1.0) & (y_e == 0.0)
        contrast_c = torch.clamp(p_e - self.margin, min=0.0).pow(2) * mask_c_only.float()

        # Case 2: Pure Effusion (y_e=1, y_c=0) -> Suppress Cardiomegaly probability
        mask_e_only = (y_e == 1.0) & (y_c == 0.0)
        contrast_e = torch.clamp(p_c - self.margin, min=0.0).pow(2) * mask_e_only.float()

        num_contrastive_samples = mask_c_only.sum() + mask_e_only.sum()
        if num_contrastive_samples > 0:
            contrast_loss = (contrast_c.sum() + contrast_e.sum()) / (num_contrastive_samples + 1e-6)
        else:
            contrast_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        return bce_loss + self.lambda_contrast * contrast_loss
