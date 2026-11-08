"""
Unit tests for DisentangledBCEWithLogitsLoss.
"""

import torch
from medicalvision.loss import DisentangledBCEWithLogitsLoss


def test_disentangled_loss_zero_when_predictions_match_targets():
    criterion = DisentangledBCEWithLogitsLoss(cardio_idx=4, eff_idx=3, margin=0.08, lambda_contrast=2.0)

    # Batch of 2:
    # 0: Pure Cardiomegaly (target=[0,0,0,0,1], pred cardio high, pred eff very low)
    # 1: Pure Effusion (target=[0,0,0,1,0], pred eff high, pred cardio very low)
    logits = torch.tensor([
        [-5.0, -5.0, -5.0, -5.0, 5.0],  # sigmoid(eff) = 0.0067 < margin
        [-5.0, -5.0, -5.0, 5.0, -5.0],  # sigmoid(cardio) = 0.0067 < margin
    ], requires_grad=True)

    targets = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])

    loss = criterion(logits, targets)
    loss.backward()

    # Contrast penalty should be strictly 0 because probabilities are below 0.08 margin
    assert loss.item() < 0.1
    assert logits.grad is not None


def test_disentangled_loss_penalizes_cross_talk():
    criterion = DisentangledBCEWithLogitsLoss(cardio_idx=4, eff_idx=3, margin=0.08, lambda_contrast=3.0)

    # Pure Cardiomegaly, but Effusion logit is high (e.g. 0.0 -> prob 0.5 > 0.08)
    logits_with_crosstalk = torch.tensor([
        [-5.0, -5.0, -5.0, 0.0, 5.0],
    ], requires_grad=True)

    logits_without_crosstalk = torch.tensor([
        [-5.0, -5.0, -5.0, -4.0, 5.0],
    ], requires_grad=True)

    targets = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    loss_crosstalk = criterion(logits_with_crosstalk, targets)
    loss_clean = criterion(logits_without_crosstalk, targets)

    # Cross talk must yield significantly higher loss
    assert loss_crosstalk.item() > loss_clean.item()


def test_comorbid_case_no_contrast_penalty():
    criterion = DisentangledBCEWithLogitsLoss(cardio_idx=4, eff_idx=3, margin=0.08, lambda_contrast=5.0)

    # Both conditions are present (e.g. CHF)
    targets_comorbid = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])
    logits_both_high = torch.tensor([[ -5.0, -5.0, -5.0, 4.0, 4.0]], requires_grad=True)

    bce_alone = torch.nn.functional.binary_cross_entropy_with_logits(logits_both_high, targets_comorbid)
    loss = criterion(logits_both_high, targets_comorbid)

    # On comorbid cases, contrast loss must be 0 (no suppression of real comorbidities)
    assert abs(loss.item() - bce_alone.item()) < 1e-4
