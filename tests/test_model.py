"""
Unit tests for MedicalVision AI model architectures and loss functions.
"""

import pytest
import torch

from medicalvision.config import NUM_CLASSES
from medicalvision.models.classifier import ChestXrayClassifier
from medicalvision.models.loss import WeightedBCELoss, MultiLabelFocalLoss


def test_densenet_forward_shape():
    model = ChestXrayClassifier(num_classes=NUM_CLASSES, backbone="densenet121", pretrained=False)
    model.eval()

    # Batch of 2 dummy images (3 channels, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)

    assert logits.shape == (2, NUM_CLASSES)


def test_resnet_and_efficientnet_forward_shape():
    for backbone in ["resnet50", "efficientnet_b0"]:
        model = ChestXrayClassifier(num_classes=NUM_CLASSES, backbone=backbone, pretrained=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        logits = model(dummy_input)
        assert logits.shape == (2, NUM_CLASSES)
        assert hasattr(model, "target_conv_layer")



def test_predict_dict_output():
    model = ChestXrayClassifier(num_classes=NUM_CLASSES, backbone="densenet121", pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    pred_dict = model.predict_dict(dummy_input)

    assert "Cardiomegaly" in pred_dict
    assert "Abnormal_Risk" in pred_dict
    assert 0.0 <= pred_dict["Cardiomegaly"] <= 1.0
    assert 0.0 <= pred_dict["Abnormal_Risk"] <= 1.0


def test_weighted_bce_loss():
    pos_weights = torch.ones(NUM_CLASSES) * 2.0
    criterion = WeightedBCELoss(pos_weight=pos_weights)

    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.randint(0, 2, (4, NUM_CLASSES)).float()

    loss = criterion(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_multilabel_focal_loss():
    criterion = MultiLabelFocalLoss(gamma=2.0, alpha=0.25)
    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.randint(0, 2, (4, NUM_CLASSES)).float()

    loss = criterion(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
