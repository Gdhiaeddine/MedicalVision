"""
Probability Calibration Engine for MedicalVision AI.
Implements Temperature Scaling (Guo et al., ICML 2017),
Expected Calibration Error (ECE), Brier Score, and Brier Skill Score.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import os

from medicalvision.config import DISEASE_CLASSES


class TemperatureScaler(nn.Module):
    """
    Applies Temperature Scaling to raw logits:
    p = sigmoid(logits / T)
    Optimizes T on validation set using Negative Log-Likelihood (BCE).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_targets: torch.Tensor, lr: float = 0.01, max_iter: int = 100):
        """
        Fit scalar temperature T on validation logits and targets.
        """
        self.temperature.data.fill_(1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        # Constrain temperature to be positive
        self.temperature.data.clamp_(min=0.1, max=10.0)
        return float(self.temperature.item())


class VectorTemperatureScaler(nn.Module):
    """
    Class-specific temperature scaling (one temperature T_c per pathology):
    p_c = sigmoid(logits_c / T_c)
    """

    def __init__(self, num_classes: int = len(DISEASE_CLASSES)):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(num_classes) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperatures.unsqueeze(0)

    def fit(self, val_logits: torch.Tensor, val_targets: torch.Tensor, lr: float = 0.05, max_iter: int = 100):
        optimizer = torch.optim.LBFGS([self.temperatures], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = criterion(scaled, val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperatures.data.clamp_(min=0.1, max=10.0)
        return self.temperatures.detach().cpu().numpy().tolist()


class PlattScaler:
    """
    Per-pathology Platt Scaling (affine calibration z_c -> w_c * z_c + b_c).
    Crucial for highly imbalanced multi-label problems to align both confidence slope
    and background prevalence, maximizing Brier Skill Score (BSS > 0) and minimizing ECE.
    """

    def __init__(self, num_classes: int = len(DISEASE_CLASSES)):
        self.num_classes = num_classes
        self.models: List[Optional[LogisticRegression]] = [None] * num_classes

    def fit(self, val_logits: np.ndarray, val_targets: np.ndarray):
        for c in range(self.num_classes):
            y_c = val_targets[:, c]
            if len(np.unique(y_c)) > 1:
                lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
                lr.fit(val_logits[:, c : c + 1], y_c)
                self.models[c] = lr
            else:
                self.models[c] = None

    def transform(self, logits: np.ndarray) -> np.ndarray:
        calib_probs = np.zeros_like(logits, dtype=np.float32)
        for c in range(self.num_classes):
            if self.models[c] is not None:
                calib_probs[:, c] = self.models[c].predict_proba(logits[:, c : c + 1])[:, 1]
            else:
                # Fallback to standard sigmoid
                calib_probs[:, c] = 1.0 / (1.0 + np.exp(-logits[:, c]))
        return calib_probs



def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE):
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_prob[in_bin])
            ece += (bin_count / n) * np.abs(acc - conf)

    return float(ece)


def compute_calibration_report(
    y_true: np.ndarray,
    y_prob_uncalibrated: np.ndarray,
    y_prob_calibrated: np.ndarray,
    disease_classes: List[str] = DISEASE_CLASSES,
) -> Dict[str, dict]:
    """
    Evaluate calibration per condition before and after temperature scaling:
    - ECE uncalibrated vs calibrated
    - Brier score vs reference prevalence Brier score
    - Brier Skill Score (BSS = 1 - Brier / Brier_ref)
    """
    report = {}

    for i, disease in enumerate(disease_classes):
        target = y_true[:, i]
        prob_uncal = y_prob_uncalibrated[:, i]
        prob_cal = y_prob_calibrated[:, i]

        # Prevalence
        p_prev = float(np.mean(target)) if len(target) > 0 else 0.0

        # Reference Brier Score (naive model predicting prevalence p_prev)
        brier_ref = float(p_prev * (1.0 - p_prev))

        # Model Brier scores
        brier_uncal = float(brier_score_loss(target, prob_uncal))
        brier_cal = float(brier_score_loss(target, prob_cal))

        # Brier Skill Score: > 0 means better than guessing prevalence
        bss = float(1.0 - (brier_cal / brier_ref)) if brier_ref > 1e-6 else 0.0

        # ECE
        ece_uncal = compute_ece(target, prob_uncal)
        ece_cal = compute_ece(target, prob_cal)

        report[disease] = {
            "Prevalence": p_prev,
            "ECE_Uncalibrated": ece_uncal,
            "ECE_Calibrated": ece_cal,
            "ECE_Improvement": float(ece_uncal - ece_cal),
            "Brier_Uncalibrated": brier_uncal,
            "Brier_Calibrated": brier_cal,
            "Brier_Reference": brier_ref,
            "Brier_Skill_Score": bss,
        }

    return report


def plot_reliability_diagrams(
    y_true: np.ndarray,
    y_prob_calibrated: np.ndarray,
    output_path: str = "outputs/reliability_diagrams_14_classes.png",
    disease_classes: List[str] = DISEASE_CLASSES,
    n_bins: int = 10,
):
    """
    Generate a 4x4 grid of reliability diagrams for all 14 diseases.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.flatten()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i, disease in enumerate(disease_classes):
        ax = axes[i]
        target = y_true[:, i]
        prob = y_prob_calibrated[:, i]

        bin_confs = []
        bin_accs = []
        bin_weights = []

        for b in range(n_bins):
            b_low = bin_boundaries[b]
            b_high = bin_boundaries[b + 1]
            in_b = (prob > b_low) & (prob <= b_high)
            count = np.sum(in_b)
            if count > 0:
                bin_accs.append(np.mean(target[in_b]))
                bin_confs.append(np.mean(prob[in_b]))
                bin_weights.append(count)

        # Plot perfect diagonal
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        if len(bin_confs) > 0:
            ax.plot(bin_confs, bin_accs, "s-", color="#1f77b4", label="Model (Calibrated)")
            ax.scatter(bin_confs, bin_accs, s=[max(20, w * 3) for w in bin_weights], color="#1f77b4", alpha=0.6)

        ece = compute_ece(target, prob, n_bins)
        p_prev = float(np.mean(target)) if len(target) > 0 else 0.0
        brier = float(brier_score_loss(target, prob))
        b_ref = p_prev * (1.0 - p_prev)
        bss = 1.0 - (brier / b_ref) if b_ref > 1e-6 else 0.0

        ax.set_title(f"{disease}\nECE: {ece:.3f} | BSS: {bss:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Mean Confidence", fontsize=9)
        ax.set_ylabel("Empirical Frequency", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide unused subplots (axes 14 and 15)
    for j in range(len(disease_classes), len(axes)):
        axes[j].axis("off")

    plt.suptitle("MedicalVision AI — Reliability Diagrams & Calibration Curves (14 Classes)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path

