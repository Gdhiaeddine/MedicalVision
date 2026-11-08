"""
Model Reliability and Uncertainty Estimation using Monte Carlo Dropout (MC Dropout).
Quantifies epistemic (model) uncertainty to flag ambiguous clinical cases.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from medicalvision.config import DISEASE_CLASSES, MC_DROPOUT_SAMPLES, HIGH_UNCERTAINTY_THRESHOLD


def enable_mc_dropout(model: nn.Module):
    """
    Enable Dropout layers during inference while keeping BatchNorm in eval mode.
    This produces stochastic forward passes representing epistemic uncertainty.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def estimate_uncertainty_mc_dropout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_samples: int = MC_DROPOUT_SAMPLES,
    disease_classes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Perform Monte Carlo Dropout sampling over T forward passes.

    Args:
        model: ChestXrayClassifier instance.
        input_tensor: Normalized image tensor (1, 3, H, W).
        num_samples: Number of stochastic forward passes (default: 10).
        disease_classes: Optional list of pathology names matching output dimension.

    Returns:
        Dictionary mapping each disease to:
        - "mean_prob": Expected predicted probability
        - "std_dev": Epistemic standard deviation (uncertainty)
        - "entropy": Binary Shannon entropy
        - "is_uncertain": Boolean flag indicating whether std exceeds threshold
    """
    model.eval()
    enable_mc_dropout(model)

    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            predictions.append(probs)

    # Convert back to standard eval mode
    model.eval()

    predictions = np.array(predictions)  # Shape: (num_samples, num_classes)
    means = np.mean(predictions, axis=0)
    stds = np.std(predictions, axis=0)

    num_out = len(means)
    if disease_classes is None:
        if hasattr(model, "classes") and model.classes and len(model.classes) == num_out:
            active_disease_classes = model.classes
        elif num_out == 5:
            from medicalvision.config_5class import FIVE_DISEASE_CLASSES
            active_disease_classes = FIVE_DISEASE_CLASSES
        elif num_out == len(DISEASE_CLASSES):
            active_disease_classes = DISEASE_CLASSES
        else:
            active_disease_classes = [f"Class_{i}" for i in range(num_out)]
    else:
        active_disease_classes = list(disease_classes)[:num_out]

    results = {}
    for idx, disease in enumerate(active_disease_classes):
        p = float(means[idx])
        # Shannon entropy
        eps = 1e-7
        entropy = - (p * np.log2(p + eps) + (1.0 - p) * np.log2(1.0 - p + eps))

        results[disease] = {
            "mean_prob": p,
            "std_dev": float(stds[idx]),
            "entropy": float(entropy),
            "is_uncertain": bool(stds[idx] > HIGH_UNCERTAINTY_THRESHOLD),
            "samples": predictions[:, idx].tolist(),
        }

    return results
