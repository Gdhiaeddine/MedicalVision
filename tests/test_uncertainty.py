"""
Unit tests for MC Dropout uncertainty estimation.
"""

import pytest
import torch

from medicalvision.config import NUM_CLASSES, DISEASE_CLASSES
from medicalvision.models.classifier import ChestXrayClassifier
from medicalvision.uncertainty import estimate_uncertainty_mc_dropout


def test_mc_dropout_uncertainty():
    model = ChestXrayClassifier(num_classes=NUM_CLASSES, backbone="densenet121", pretrained=False, dropout_rate=0.3)
    dummy_input = torch.randn(1, 3, 224, 224)

    results = estimate_uncertainty_mc_dropout(model, dummy_input, num_samples=5)

    assert len(results) == len(DISEASE_CLASSES)
    for disease in DISEASE_CLASSES:
        res = results[disease]
        assert "mean_prob" in res
        assert "std_dev" in res
        assert "entropy" in res
        assert 0.0 <= res["mean_prob"] <= 1.0
        assert res["std_dev"] >= 0.0
        assert len(res["samples"]) == 5


def test_mc_dropout_uncertainty_5class():
    from medicalvision.config_5class import FIVE_DISEASE_CLASSES

    model = ChestXrayClassifier(num_classes=5, backbone="densenet121", pretrained=False, dropout_rate=0.3)
    dummy_input = torch.randn(1, 3, 224, 224)

    results = estimate_uncertainty_mc_dropout(model, dummy_input, num_samples=5, disease_classes=FIVE_DISEASE_CLASSES)

    assert len(results) == 5
    for disease in FIVE_DISEASE_CLASSES:
        res = results[disease]
        assert "mean_prob" in res
        assert "std_dev" in res
        assert 0.0 <= res["mean_prob"] <= 1.0
