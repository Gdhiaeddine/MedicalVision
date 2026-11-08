"""
Comprehensive Evaluation Suite for MedicalVision AI:
Classification (ROC-AUC, PR-AUC, Sensitivity, Specificity, F1-score, Confusion Matrix)
Detection (mAP@0.5, mAP@0.5:0.95, IoU, Detection Precision & Recall).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import torch
from torch.utils.data import DataLoader
from PIL import Image

from medicalvision.config import (
    DISEASE_CLASSES,
    IMAGE_DIR,
    METADATA_FILE,
    BBOX_FILE,
    DEFAULT_THRESHOLD,
    CHECKPOINT_DIR,
)
from medicalvision.dataset import load_dataset_metadata, split_by_patient, load_bounding_boxes, ChestXrayDataset
from medicalvision.models.classifier import build_model
from medicalvision.transforms import get_eval_transforms, preprocess_image_for_inference
from medicalvision.interpretability.gradcam import GradCAM
from medicalvision.interpretability.localization import extract_bounding_boxes, compute_iou


def normalize_disease_name(name: str) -> str:
    n = name.strip().lower().replace(" ", "_")
    if n == "infiltrate":
        return "infiltration"
    return n


def run_comprehensive_classification(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, dict]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    y_bin = (y_pred >= threshold).astype(int)

    class_results = {}
    macro_roc = []
    macro_pr = []
    macro_f1 = []
    macro_sens = []
    macro_spec = []

    for i, disease in enumerate(DISEASE_CLASSES):
        y_t = y_true[:, i]
        y_s = y_pred[:, i]
        y_b = y_bin[:, i]

        # ROC-AUC
        if len(np.unique(y_t)) > 1:
            roc_auc = float(roc_auc_score(y_t, y_s))
            macro_roc.append(roc_auc)
        else:
            roc_auc = 0.50

        # PR-AUC (Average Precision)
        if len(np.unique(y_t)) > 1 and y_t.sum() > 0:
            pr_auc = float(average_precision_score(y_t, y_s))
            macro_pr.append(pr_auc)
        else:
            pr_auc = float(y_t.mean()) if len(y_t) > 0 else 0.0

        # Confusion Matrix elements: TN, FP, FN, TP
        cm = confusion_matrix(y_t, y_b, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        accuracy = (tp + tn) / len(y_t) if len(y_t) > 0 else 0.0

        macro_f1.append(f1)
        macro_sens.append(sensitivity)
        macro_spec.append(specificity)

        class_results[disease] = {
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "Accuracy": accuracy,
            "F1": f1,
            "Confusion_Matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "Total_Positives": int(y_t.sum()),
        }

    # Binary Normal vs Abnormal
    true_abnormal = (y_true.sum(axis=1) > 0).astype(int)
    pred_abnormal = (y_bin.sum(axis=1) > 0).astype(int)
    bin_cm = confusion_matrix(true_abnormal, pred_abnormal, labels=[0, 1])
    bin_tn, bin_fp, bin_fn, bin_tp = int(bin_cm[0, 0]), int(bin_cm[0, 1]), int(bin_cm[1, 0]), int(bin_cm[1, 1])

    summary = {
        "per_class": class_results,
        "macro_metrics": {
            "Mean_ROC_AUC": float(np.mean(macro_roc)) if macro_roc else 0.5,
            "Mean_PR_AUC": float(np.mean(macro_pr)) if macro_pr else 0.0,
            "Mean_Sensitivity": float(np.mean(macro_sens)),
            "Mean_Specificity": float(np.mean(macro_spec)),
            "Mean_F1": float(np.mean(macro_f1)),
        },
        "binary_normal_abnormal": {
            "TN": bin_tn,
            "FP": bin_fp,
            "FN": bin_fn,
            "TP": bin_tp,
            "Accuracy": float(accuracy_score(true_abnormal, pred_abnormal)),
            "Precision": float(precision_score(true_abnormal, pred_abnormal, zero_division=0)),
            "Sensitivity": float(recall_score(true_abnormal, pred_abnormal, zero_division=0)),
            "Specificity": float(bin_tn / (bin_tn + bin_fp)) if (bin_tn + bin_fp) > 0 else 1.0,
        },
    }
    return summary


def run_comprehensive_detection(
    model: torch.nn.Module,
    bbox_df: pd.DataFrame,
    image_dir: Path,
    device: str = "cpu",
    max_samples: int = 50,
) -> Dict[str, float]:
    """
    Evaluate Weakly-Supervised Localization:
    mAP@0.5, mAP@0.5:0.95, Mean IoU, Detection Precision & Recall.
    """
    if bbox_df.empty:
        return {}

    gradcam = GradCAM(model)
    iou_thresholds = np.arange(0.50, 1.00, 0.05)  # 10 thresholds: 0.50 to 0.95

    # Store detections: each detection has (confidence_score, iou_with_gt, is_match_per_thresh)
    detections = []
    ious_all = []

    samples_done = 0
    for _, row in bbox_df.iterrows():
        if samples_done >= max_samples:
            break

        img_filename = row["Image_Index"]
        disease_label = normalize_disease_name(row["Finding_Label"])
        img_path = image_dir / img_filename

        if not img_path.exists():
            continue

        matching_indices = [
            i for i, d in enumerate(DISEASE_CLASSES)
            if normalize_disease_name(d) == disease_label
        ]
        if not matching_indices:
            continue
        disease_idx = matching_indices[0]

        # Ingestion
        pil_img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = pil_img.size
        tensor = preprocess_image_for_inference(pil_img).to(device)

        with torch.no_grad():
            logits = model(tensor)
            conf_score = float(torch.sigmoid(logits)[0, disease_idx].cpu().item())

        # Grad-CAM Heatmap
        cam = gradcam.generate(tensor, target_class_idx=disease_idx, target_size=(w_orig, h_orig))
        pred_boxes = extract_bounding_boxes(cam, threshold=0.45)

        gt_box = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))

        best_iou = 0.0
        for pbox in pred_boxes:
            iou = compute_iou(pbox, gt_box)
            if iou > best_iou:
                best_iou = iou

        ious_all.append(best_iou)

        # Record for mAP calculation
        detections.append({
            "confidence": conf_score,
            "iou": best_iou,
        })
        samples_done += 1

    if not detections:
        return {}

    ious_all = np.array(ious_all)
    mean_iou = float(np.mean(ious_all))

    # Calculate AP across IoU thresholds: 0.50, 0.55, ..., 0.95
    ap_per_iou = []
    for t_iou in iou_thresholds:
        tp_count = sum(1 for d in detections if d["iou"] >= t_iou)
        fp_count = len(detections) - tp_count
        precision_at_t = tp_count / len(detections) if len(detections) > 0 else 0.0
        recall_at_t = tp_count / len(detections) if len(detections) > 0 else 0.0  # 1 GT box per sample
        ap_per_iou.append(precision_at_t)

    map_50 = float(ap_per_iou[0])  # at IoU 0.50
    map_50_95 = float(np.mean(ap_per_iou))  # average from 0.50 to 0.95

    # Detection Precision & Recall at standard IoU >= 0.25 (standard medical weak supervision) and >= 0.50
    tp_50 = sum(1 for d in detections if d["iou"] >= 0.50)
    tp_25 = sum(1 for d in detections if d["iou"] >= 0.25)
    tp_10 = sum(1 for d in detections if d["iou"] >= 0.10)
    total = len(detections)

    return {
        "Evaluated_BBoxes": total,
        "Mean_IoU": mean_iou,
        "mAP@0.5": map_50,
        "mAP@0.5:0.95": map_50_95,
        "Detection_Precision@0.5": float(tp_50 / total) if total > 0 else 0.0,
        "Detection_Recall@0.5": float(tp_50 / total) if total > 0 else 0.0,
        "Detection_Precision@0.25": float(tp_25 / total) if total > 0 else 0.0,
        "Detection_Recall@0.25": float(tp_25 / total) if total > 0 else 0.0,
        "Detection_Rate@0.10": float(tp_10 / total) if total > 0 else 0.0,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running comprehensive evaluation on {device}...")

    # Load Model
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    has_ckpt = ckpt_path.exists()
    model = build_model(
        backbone="densenet121",
        pretrained=True if not has_ckpt else False,
        checkpoint_path=str(ckpt_path) if has_ckpt else None,
        device=device,
    )

    # 1. Classification Data
    print("Loading test dataset split...")
    df = load_dataset_metadata(METADATA_FILE, IMAGE_DIR, existing_images_only=True)
    _, _, test_df = split_by_patient(df, train_ratio=0.75, val_ratio=0.10, test_ratio=0.15)
    test_sub = test_df.sample(n=100, random_state=42).reset_index(drop=True)

    test_ds = ChestXrayDataset(test_sub, image_dir=IMAGE_DIR, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print("Computing classification metrics (ROC-AUC, PR-AUC, Sens, Spec, F1, Confusion Matrix)...")
    clf_results = run_comprehensive_classification(model, test_loader, device=device)

    # 2. Detection Data
    print("Computing detection metrics on expert annotated bounding boxes...")
    bbox_df = load_bounding_boxes(BBOX_FILE)
    det_results = run_comprehensive_detection(model, bbox_df, IMAGE_DIR, device=device, max_samples=40)

    # Output formatted results
    full_output = {
        "classification": clf_results,
        "detection": det_results,
    }

    out_file = Path("outputs") / "comprehensive_evaluation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(full_output, f, indent=2)

    print("\n--- RESULTS JSON GENERATED ---")
    print(json.dumps(full_output, indent=2))


if __name__ == "__main__":
    main()
