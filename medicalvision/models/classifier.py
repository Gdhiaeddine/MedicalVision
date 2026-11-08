"""
Deep Learning Classifier architectures (DenseNet-121 / CheXNet and ResNet-50)
for 14-pathology multi-label chest X-ray diagnosis.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models

from medicalvision.config import DISEASE_CLASSES, DROPOUT_RATE, NUM_CLASSES


class ChestXrayClassifier(nn.Module):
    """
    Multi-label Chest X-ray Classifier supporting DenseNet-121 (CheXNet standard)
    and ResNet-50 backbones.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone.lower()
        self.dropout_rate = dropout_rate

        if self.backbone_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            base_model = models.densenet121(weights=weights)
            self.features = base_model.features
            in_features = base_model.classifier.in_features
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(in_features, num_classes)
            self.target_conv_layer = self.features.denseblock4.denselayer16.conv2

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            # Retain conv layers up to layer4
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool
            in_features = base_model.fc.in_features
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(in_features, num_classes)
            self.target_conv_layer = self.layer4[-1].conv3

        elif self.backbone_name in ["efficientnet_b0", "efficientnet-b0", "efficientnetb0"]:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            in_features = base_model.classifier[1].in_features
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(in_features, num_classes)
            self.target_conv_layer = self.features[-1][0]

        elif self.backbone_name in ["convnext_tiny", "convnext"]:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base_model = models.convnext_tiny(weights=weights)
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            in_features = base_model.classifier[2].in_features
            self.dropout = nn.Dropout(p=dropout_rate)
            self.classifier = nn.Linear(in_features, num_classes)
            self.target_conv_layer = self.features[-1][-1].block[0]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'densenet121', 'resnet50', 'efficientnet_b0', or 'convnext_tiny'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits (shape: [batch_size, num_classes]).
        """
        if self.backbone_name == "densenet121":
            features = self.features(x)
            out = nn.functional.relu(features, inplace=True)
            out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.dropout(out)
            logits = self.classifier(out)
            return logits

        elif self.backbone_name == "resnet50":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits

        elif self.backbone_name in ["efficientnet_b0", "efficientnet-b0", "efficientnetb0"]:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits

        elif self.backbone_name in ["convnext_tiny", "convnext"]:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return calibrated probabilities in [0, 1] using Sigmoid activation.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def predict_dict(self, x: torch.Tensor, disease_classes=DISEASE_CLASSES) -> Dict[str, float]:
        """
        Predict probability for a single image tensor and return a clean dictionary.
        Also calculates an overall abnormality risk score.
        """
        probs = self.predict_proba(x)[0].cpu().numpy()
        result = {disease: float(probs[i]) for i, disease in enumerate(disease_classes)}
        # Aggregated Thoracic Abnormality Score: 1 - product of (1 - p_i)
        # i.e., probability that at least one thoracic condition is present
        prob_normal = float(torch.prod(1.0 - torch.tensor(probs)).item())
        result["Abnormal_Risk"] = 1.0 - prob_normal
        result["Is_Abnormal"] = result["Abnormal_Risk"] >= 0.5
        return result


def build_model(
    backbone: str = "densenet121",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> ChestXrayClassifier:
    """
    Factory function to initialize and configure model with optional weights.
    Supports both raw state_dict and checkpoint dictionaries with 'model_state_dict',
    with automatic backbone architecture detection from checkpoint keys.
    """
    state_dict = None
    if checkpoint_path and os.path.exists(str(checkpoint_path)):
        map_loc = device if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"
        checkpoint = torch.load(str(checkpoint_path), map_location=map_loc, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if "backbone" in checkpoint:
                backbone = checkpoint["backbone"]
            if "classes" in checkpoint and len(checkpoint["classes"]) > 0:
                num_classes = len(checkpoint["classes"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Auto-detect backbone architecture from keys and weight shapes if needed
        if state_dict is not None and isinstance(state_dict, dict):
            keys = list(state_dict.keys())
            if any(k.startswith("features.denseblock") for k in keys):
                backbone = "densenet121"
            elif any(k.startswith("layer1.") for k in keys):
                backbone = "resnet50"
            elif any(k.endswith(".layer_scale") for k in keys) or (
                "features.0.0.weight" in state_dict and state_dict["features.0.0.weight"].shape[0] == 96
            ):
                backbone = "convnext_tiny"
            elif (
                "features.0.0.weight" in state_dict and state_dict["features.0.0.weight"].shape[0] == 32
            ) or any(k.endswith(".num_batches_tracked") for k in keys):
                backbone = "efficientnet_b0"

    model = ChestXrayClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained if (state_dict is None) else False,
    )

    if state_dict is not None:
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            cleaned_state_dict[new_key] = v

        model.load_state_dict(cleaned_state_dict)

    model.to(device)
    return model
