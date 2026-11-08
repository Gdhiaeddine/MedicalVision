"""
Explainability and weakly supervised localization modules.
"""

from medicalvision.interpretability.gradcam import GradCAM, overlay_heatmap_on_image
from medicalvision.interpretability.localization import extract_bounding_boxes, compute_iou

__all__ = [
    "GradCAM",
    "overlay_heatmap_on_image",
    "extract_bounding_boxes",
    "compute_iou",
]
