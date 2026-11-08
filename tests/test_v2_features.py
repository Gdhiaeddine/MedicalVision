"""
Unit tests for MedicalVision AI v2.0 features:
- Binary screening engine
- Multi-label threshold optimization
- Platt calibration and BSS calculation
- Faster R-CNN object detector creation and box transformation
"""

import numpy as np
import pytest
import torch

from medicalvision.binary_screener import BinaryScreener
from medicalvision.thresholding import find_optimal_thresholds, compute_metrics_at_threshold
from medicalvision.calibration import PlattScaler, compute_ece
from medicalvision.models.detector import create_faster_rcnn_detector
from medicalvision.bootstrap import paired_patient_bootstrap


def test_binary_screener_fit_predict():
    screener = BinaryScreener(target_specificity=0.80)
    # Synthetic logits and binary targets
    logits = np.random.randn(100, 14)
    y_binary = np.random.binomial(1, 0.4, 100)

    screener.fit_meta_model(logits, y_binary)
    preds, probs = screener.predict_abnormal(logits=logits)

    assert len(preds) == 100
    assert len(probs) == 100
    assert set(np.unique(preds)).issubset({0, 1})
    assert 0.0 <= screener.optimal_threshold <= 1.0


def test_thresholding_strategies():
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.7, 0.15, 0.25, 0.85])

    thresholds = find_optimal_thresholds(y_true, y_probs)
    assert "youden" in thresholds
    assert "f1" in thresholds
    assert "f2" in thresholds
    assert "constrained_spec75" in thresholds

    metrics = compute_metrics_at_threshold(y_true, y_probs, thresholds["constrained_spec75"])
    assert "precision" in metrics
    assert "recall" in metrics
    assert "specificity" in metrics
    assert metrics["specificity"] >= 0.70


def test_platt_scaler():
    scaler = PlattScaler(num_classes=2)
    val_logits = np.random.randn(100, 2)
    val_targets = np.random.binomial(1, 0.1, (100, 2))

    scaler.fit(val_logits, val_targets)
    probs = scaler.transform(val_logits)

    assert probs.shape == (100, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    ece = compute_ece(val_targets[:, 0], probs[:, 0])
    assert 0.0 <= ece <= 1.0


def test_faster_rcnn_detector_creation():
    model = create_faster_rcnn_detector(num_classes=9, pretrained_backbone=False)
    model.eval()
    dummy_img = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        preds = model(dummy_img)
    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "scores" in preds[0]
    assert "labels" in preds[0]


def test_paired_patient_bootstrap():
    patient_ids = np.array(["P1", "P1", "P2", "P3", "P4", "P4", "P5", "P6", "P7", "P8"])
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    prob_14 = np.array([0.1, 0.15, 0.6, 0.2, 0.4, 0.5, 0.1, 0.3, 0.5, 0.2])
    prob_5 = np.array([0.05, 0.1, 0.8, 0.1, 0.7, 0.75, 0.05, 0.2, 0.8, 0.1])

    res = paired_patient_bootstrap(
        patient_ids=patient_ids,
        y_true=y_true,
        prob_14=prob_14,
        prob_5=prob_5,
        threshold_14=0.5,
        threshold_5=0.5,
        repetitions=100,
        seed=42,
    )

    assert "metrics" in res
    assert "sensitivity" in res["metrics"]
    assert "model_14" in res["metrics"]["sensitivity"]
    assert "model_5" in res["metrics"]["sensitivity"]
    assert "difference" in res["metrics"]["sensitivity"]
    assert "ci_95" in res["metrics"]["sensitivity"]["difference"]

