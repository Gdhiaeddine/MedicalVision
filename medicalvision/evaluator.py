"""
Evaluation and benchmarking suite for MedicalVision AI.
Calculates multi-label diagnostic metrics (AUROC, Sensitivity, Specificity, F1)
and weakly supervised bounding box localization IoU.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss,
)
from PIL import Image

from medicalvision.config import DISEASE_CLASSES, DEFAULT_THRESHOLD
from medicalvision.transforms import preprocess_image_for_inference
from medicalvision.interpretability.gradcam import GradCAM
from medicalvision.interpretability.localization import extract_bounding_boxes, compute_iou


def evaluate_dataset_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    disease_classes: List[str] = DISEASE_CLASSES,
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics across all 14 conditions:
    Accuracy, Precision, Recall/Sensitivity, Specificity, F1, AUROC,
    as well as Hamming Accuracy and Normal vs Abnormal Binary metrics.
    """
    results = {}
    auroc_list = []
    f1_list = []
    precision_list = []
    accuracy_list = []

    y_pred_bin = (y_pred_probs >= threshold).astype(int)

    for i, disease in enumerate(disease_classes):
        target = y_true[:, i]
        scores = y_pred_probs[:, i]
        pred_b = y_pred_bin[:, i]

        # Per-class Accuracy: (TP + TN) / Total
        acc = float(accuracy_score(target, pred_b))
        accuracy_list.append(acc)

        # Per-class Precision: TP / (TP + FP)
        prec = float(precision_score(target, pred_b, zero_division=0))
        precision_list.append(prec)

        # AUROC
        if len(np.unique(target)) > 1:
            auc = float(roc_auc_score(target, scores))
            auroc_list.append(auc)
        else:
            auc = 0.5

        # Sensitivity (Recall) & Specificity
        pos_indices = np.where(target == 1)[0]
        neg_indices = np.where(target == 0)[0]

        sensitivity = float(pred_b[pos_indices].mean()) if len(pos_indices) > 0 else 0.0
        specificity = float((1 - pred_b[neg_indices]).mean()) if len(neg_indices) > 0 else 1.0

        f1 = float(f1_score(target, pred_b, zero_division=0))
        f1_list.append(f1)

        results[disease] = {
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1": f1,
            "AUROC": auc,
            "Positives": int(target.sum()),
        }

    # Aggregate Multi-label Metrics
    results["Macro_Accuracy"] = float(np.mean(accuracy_list))
    results["Hamming_Accuracy"] = float(1.0 - hamming_loss(y_true, y_pred_bin))
    results["Subset_Exact_Accuracy"] = float((y_true == y_pred_bin).all(axis=1).mean())

    results["Macro_Precision"] = float(precision_score(y_true, y_pred_bin, average="macro", zero_division=0))
    results["Micro_Precision"] = float(precision_score(y_true, y_pred_bin, average="micro", zero_division=0))
    results["Macro_F1"] = float(np.mean(f1_list)) if f1_list else 0.0
    results["Macro_AUROC"] = float(np.mean(auroc_list)) if auroc_list else 0.5

    # Normal vs Abnormal Binary Detection
    true_is_abnormal = (y_true.sum(axis=1) > 0).astype(int)
    pred_is_abnormal = (y_pred_bin.sum(axis=1) > 0).astype(int)

    results["Binary_Abnormal_Accuracy"] = float(accuracy_score(true_is_abnormal, pred_is_abnormal))
    results["Binary_Abnormal_Precision"] = float(precision_score(true_is_abnormal, pred_is_abnormal, zero_division=0))
    results["Binary_Abnormal_Recall"] = float(recall_score(true_is_abnormal, pred_is_abnormal, zero_division=0))

    return results


def evaluate_localization_iou(
    model: nn.Module,
    bbox_df: pd.DataFrame,
    image_dir: Path,
    device: str = "cpu",
    cam_threshold: float = 0.5,
    max_samples: int = 50,
) -> Dict[str, float]:
    """
    Evaluate weakly-supervised localization against expert ground-truth bounding boxes.
    Computes Mean IoU and Detection Rate (IoU >= 0.10 and >= 0.25).
    """
    if bbox_df.empty:
        return {"Mean_IoU": 0.0, "Detection_Rate_0.10": 0.0, "Detection_Rate_0.25": 0.0}

    gradcam = GradCAM(model)
    ious = []

    samples_processed = 0
    for _, row in bbox_df.iterrows():
        if samples_processed >= max_samples:
            break

        img_filename = row["Image_Index"]
        disease_label = row["Finding_Label"]
        img_path = image_dir / img_filename

        if not img_path.exists():
            continue

        # Match disease class index
        matching_indices = [
            i for i, d in enumerate(DISEASE_CLASSES)
            if d.lower() == disease_label.lower() or d.replace("_", " ").lower() == disease_label.replace("_", " ").lower()
        ]
        if not matching_indices:
            continue
        disease_idx = matching_indices[0]

        # Load image & run Grad-CAM
        pil_img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = pil_img.size
        tensor = preprocess_image_for_inference(pil_img).to(device)

        cam = gradcam.generate(tensor, target_class_idx=disease_idx, target_size=(w_orig, h_orig))

        # Extract predicted bounding box
        pred_boxes = extract_bounding_boxes(cam, threshold=cam_threshold)

        # Ground truth box: [x, y, w, h]
        gt_box = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))

        best_iou = 0.0
        for pbox in pred_boxes:
            iou = compute_iou(pbox, gt_box)
            if iou > best_iou:
                best_iou = iou

        ious.append(best_iou)
        samples_processed += 1

    if not ious:
        return {"Mean_IoU": 0.0, "Detection_Rate_0.10": 0.0, "Detection_Rate_0.25": 0.0}

    ious = np.array(ious)
    return {
        "Mean_IoU": float(np.mean(ious)),
        "Detection_Rate_0.10": float((ious >= 0.10).mean()),
        "Detection_Rate_0.25": float((ious >= 0.25).mean()),
        "Evaluated_Samples": len(ious),
    }
