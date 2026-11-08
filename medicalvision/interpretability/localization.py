"""
Weakly Supervised Bounding Box Localization from Grad-CAM Heatmaps.
Enables suspicious region detection without requiring bounding box training supervision.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw


def extract_bounding_boxes(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_area_ratio: float = 0.01,
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes [x, y, w, h] from high-activation regions of a 2D Grad-CAM heatmap.
    Uses connected component labeling via simple scipy/numpy contouring.

    Args:
        heatmap: 2D array [0.0, 1.0] representing normalized CAM.
        threshold: Heatmap intensity threshold (e.g. 0.5 * max).
        min_area_ratio: Minimum bounding box area as fraction of total image area.

    Returns:
        List of bounding boxes in format: [(x, y, w, h), ...]
    """
    h, w = heatmap.shape
    binary_mask = (heatmap >= threshold).astype(np.uint8)

    boxes = []
    total_area = h * w

    # Lightweight connected component analysis without OpenCV dependency
    from scipy.ndimage import label, find_objects

    labeled_array, num_features = label(binary_mask)
    slices = find_objects(labeled_array)

    for slc in slices:
        y_slice, x_slice = slc
        box_y = y_slice.start
        box_x = x_slice.start
        box_h = y_slice.stop - y_slice.start
        box_w = x_slice.stop - x_slice.start

        area = box_w * box_h
        if area >= total_area * min_area_ratio:
            boxes.append((int(box_x), int(box_y), int(box_w), int(box_h)))

    # If no region passed threshold, take bounding box around peak region
    if not boxes and heatmap.max() > 0.2:
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        box_size = int(min(h, w) * 0.2)
        box_x = max(0, peak_x - box_size // 2)
        box_y = max(0, peak_y - box_size // 2)
        box_w = min(w - box_x, box_size)
        box_h = min(h - box_y, box_size)
        boxes.append((int(box_x), int(box_y), int(box_w), int(box_h)))

    return boxes


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format: (x, y, w, h).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def draw_bounding_boxes_on_image(
    image: Image.Image,
    predicted_boxes: List[Tuple[int, int, int, int]],
    ground_truth_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    pred_label: str = "Prediction",
    gt_label: str = "Ground Truth",
) -> Image.Image:
    """
    Overlay predicted (Red/Yellow) and ground-truth (Green) bounding boxes onto an image.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Draw Ground Truth in Green
    if ground_truth_boxes:
        for box in ground_truth_boxes:
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)
            draw.text((x + 4, max(0, y - 14)), f"GT: {gt_label}", fill="lime")

    # Draw Predictions in Red/Amber
    for box in predicted_boxes:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        draw.text((x + 4, max(0, y - 14)), f"Pred: {pred_label}", fill="red")

    return annotated
