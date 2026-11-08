"""
MedicalVision AI - Detection Evaluation Engine.
Computes IoU, Average Precision (AP@0.5 and AP@0.5:0.95), and mAP
for multi-class bounding box object detection according to standard PASCAL VOC & COCO protocols.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_ap(
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[float, int, int, int]:
    """
    Computes Average Precision at a specific IoU threshold using 11-point or all-point interpolation.
    Returns: (AP, tp_count, fp_count, fn_count)
    """
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return (0.0, 0, len(pred_boxes), 0)
    if len(pred_boxes) == 0:
        return (0.0, 0, 0, n_gt)

    # Sort predictions by descending score
    sort_indices = np.argsort(-np.array(pred_scores))
    sorted_boxes = [pred_boxes[i] for i in sort_indices]

    tp = np.zeros(len(sorted_boxes))
    fp = np.zeros(len(sorted_boxes))
    gt_matched = [False] * n_gt

    for p_idx, p_box in enumerate(sorted_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g_box in enumerate(gt_boxes):
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
            tp[p_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[p_idx] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / n_gt

    # Standard VOC-style 11-point interpolation or AUC under PR curve
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 11):
        prec_at_r = precisions[recalls >= r]
        max_p = np.max(prec_at_r) if len(prec_at_r) > 0 else 0.0
        ap += max_p / 11.0

    total_tp = int(np.sum(tp))
    total_fp = int(np.sum(fp))
    total_fn = int(n_gt - total_tp)

    return float(ap), total_tp, total_fp, total_fn


def evaluate_detection_predictions(
    all_gt: Dict[str, Dict[str, List[np.ndarray]]],
    all_preds: Dict[str, Dict[str, List[Tuple[np.ndarray, float]]]],
    classes: List[str],
) -> Dict:
    """
    Evaluates mAP@0.5 and mAP@0.5:0.95 across all test images and classes.
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    per_class_results = {}
    aps_at_50 = []
    aps_coco = []

    for c in classes:
        # Collect all predictions and ground truths for this class
        class_pred_boxes = []
        class_pred_scores = []
        class_gt_boxes = []

        # We must index per image to match properly
        img_names = set(list(all_gt.keys()) + list(all_preds.keys()))

        # Evaluate across the 10 IoU thresholds for COCO mAP
        threshold_aps = []
        tp_50, fp_50, fn_50 = 0, 0, 0

        for t_idx, iou_t in enumerate(iou_thresholds):
            t_tp, t_fp, t_fn = 0, 0, 0
            img_aps = []

            for img in img_names:
                g_boxes = all_gt.get(img, {}).get(c, [])
                p_items = all_preds.get(img, {}).get(c, [])
                p_boxes = [item[0] for item in p_items]
                p_scores = [item[1] for item in p_items]

                ap, tp, fp, fn = compute_ap(p_boxes, p_scores, g_boxes, iou_threshold=iou_t)
                if len(g_boxes) > 0:
                    img_aps.append(ap)
                t_tp += tp
                t_fp += fp
                t_fn += fn

            mean_t_ap = float(np.mean(img_aps)) if len(img_aps) > 0 else 0.0
            threshold_aps.append(mean_t_ap)

            if np.isclose(iou_t, 0.5):
                tp_50 = t_tp
                fp_50 = t_fp
                fn_50 = t_fn

        ap_50 = threshold_aps[0]
        ap_coco = float(np.mean(threshold_aps))

        aps_at_50.append(ap_50)
        aps_coco.append(ap_coco)

        prec_50 = float(tp_50 / (tp_50 + fp_50)) if (tp_50 + fp_50) > 0 else 0.0
        rec_50 = float(tp_50 / (tp_50 + fn_50)) if (tp_50 + fn_50) > 0 else 0.0

        per_class_results[c] = {
            "ap_at_50": round(ap_50, 4),
            "ap_coco_50_95": round(ap_coco, 4),
            "precision_at_50": round(prec_50, 4),
            "recall_at_50": round(rec_50, 4),
            "tp": tp_50,
            "fp": fp_50,
            "fn": fn_50,
        }

    return {
        "mAP_at_50": round(float(np.mean(aps_at_50)), 4),
        "mAP_at_50_95": round(float(np.mean(aps_coco)), 4),
        "per_class": per_class_results,
    }
