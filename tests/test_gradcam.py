"""
Unit tests for Grad-CAM explainability and bounding box extraction.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from medicalvision.config import NUM_CLASSES
from medicalvision.models.classifier import ChestXrayClassifier
from medicalvision.interpretability.gradcam import GradCAM, overlay_heatmap_on_image
from medicalvision.interpretability.localization import extract_bounding_boxes, compute_iou


def test_gradcam_generation():
    model = ChestXrayClassifier(num_classes=NUM_CLASSES, backbone="densenet121", pretrained=False)
    gradcam = GradCAM(model)

    dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = gradcam.generate(dummy_tensor, target_class_idx=1)

    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0


def test_overlay_heatmap():
    orig_img = Image.new("RGB", (256, 256), color="gray")
    heatmap = np.random.rand(224, 224).astype(np.float32)

    blended = overlay_heatmap_on_image(orig_img, heatmap, alpha=0.5)
    assert blended.size == (256, 256)
    assert blended.mode == "RGB"


def test_compute_iou():
    boxA = (10, 10, 50, 50)
    boxB = (10, 10, 50, 50)
    assert compute_iou(boxA, boxB) == 1.0

    boxC = (100, 100, 50, 50)
    assert compute_iou(boxA, boxC) == 0.0

    # Half overlap
    boxD = (10, 10, 50, 25)
    assert 0.4 < compute_iou(boxA, boxD) < 0.6
