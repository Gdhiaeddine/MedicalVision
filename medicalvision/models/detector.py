"""
MedicalVision AI - Faster R-CNN Object Detection Model for Chest Pathology.
Uses ResNet-50-FPN backbone with custom FastRCNNPredictor head.
"""

from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from medicalvision.config import DETECTION_NUM_CLASSES, DETECTION_CLASSES, CHECKPOINT_DIR
from medicalvision.detection_dataset import IDX_TO_CLASS


def create_faster_rcnn_detector(
    num_classes: int = DETECTION_NUM_CLASSES,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    """
    Creates Faster R-CNN with ResNet-50-FPN backbone.
    Replaces the classification/box regression head for our 8 pathologies (+ background).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Get input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the box predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class CXRDetector:
    """High-level wrapper for running inference with Faster R-CNN."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        confidence_threshold: float = 0.25,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.model = create_faster_rcnn_detector(pretrained_backbone=False)

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> List[Dict]:
        """
        Runs object detection on single 3-channel image tensor [3, H, W] in [0, 1].
        Returns list of detected boxes:
        [{'label': str, 'score': float, 'box': [x1, y1, x2, y2]}]
        """
        self.model.eval()
        if image_tensor.dim() == 3:
            images = [image_tensor.to(self.device)]
        else:
            images = [img.to(self.device) for img in image_tensor]

        predictions = self.model(images)
        pred = predictions[0]

        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        results = []
        for box, score, label_idx in zip(boxes, scores, labels):
            if score >= self.confidence_threshold and label_idx in IDX_TO_CLASS:
                results.append({
                    "label": IDX_TO_CLASS[label_idx],
                    "score": round(float(score), 4),
                    "box": [round(float(c), 1) for c in box],
                })

        return results
