"""
Final Unseen Test Evaluation with Validation-Calibrated Decision Thresholds.
Evaluates:
- Classification: ROC-AUC, PR-AUC, Sensitivity, Specificity, F1-Score, Confusion Matrix, ECE, Brier Score.
- Detection: mAP@0.5, mAP@0.5:0.95, Mean IoU, Precision, Recall.
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
    brier_score_loss,
)
import torch
from torch.utils.data import DataLoader
from PIL import Image

from medicalvision.config import (
    DISEASE_CLASSES,
    IMAGE_DIR,
    METADATA_FILE,
    BBOX_FILE,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
)
from medicalvision.dataset import load_dataset_metadata, split_by_patient, load_bounding_boxes, ChestXrayDataset
from medicalvision.models.classifier import build_model
from medicalvision.transforms import get_eval_transforms, preprocess_image_for_inference
from medicalvision.interpretability.gradcam import GradCAM
from medicalvision.interpretability.localization import extract_bounding_boxes, compute_iou


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return float(ece)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  FINAL EVALUATION ON UNSEEN TEST SET (CALIBRATED THRESHOLDS)")
    print("=" * 70)

    # 1. Load trained checkpoint
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    assert ckpt_path.exists(), "Trained checkpoint best_model.pth not found!"
    print(f"Loading checkpoint: {ckpt_path}")
    model = build_model(backbone="densenet121", pretrained=False, checkpoint_path=str(ckpt_path), device=device)
    model.eval()

    # 2. Load optimal thresholds found on validation set
    thresh_file = OUTPUT_DIR / "optimal_thresholds.json"
    if thresh_file.exists():
        with open(thresh_file, "r") as f:
            optimal_thresholds = json.load(f)
        print("Loaded validation-calibrated thresholds.")
    else:
        print("Warning: optimal_thresholds.json not found, using 0.50 default.")
        optimal_thresholds = {d: 0.50 for d in DISEASE_CLASSES}

    # 3. Load independent Test Set
    print("\n[Loading unseen test cohort...]")
    df = load_dataset_metadata(METADATA_FILE, IMAGE_DIR, existing_images_only=True)
    _, _, test_df = split_by_patient(df, train_ratio=0.75, val_ratio=0.10, test_ratio=0.15)
    test_sub = test_df.sample(n=min(len(test_df), 300), random_state=42).reset_index(drop=True)
    print(f"Evaluating on {len(test_sub):,} unseen test X-rays (zero patient leakage).")

    test_ds = ChestXrayDataset(test_sub, image_dir=IMAGE_DIR, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

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

    # Compute Classification Metrics
    results_clf = {}
    macro_roc = []
    macro_pr = []
    macro_f1 = []
    macro_sens = []
    macro_spec = []
    macro_prec = []
    macro_ece = []
    macro_brier = []

    y_bin_calibrated = np.zeros_like(y_pred, dtype=int)

    for i, disease in enumerate(DISEASE_CLASSES):
        y_t = y_true[:, i]
        y_s = y_pred[:, i]
        tau = optimal_thresholds.get(disease, 0.50)
        y_b = (y_s >= tau).astype(int)
        y_bin_calibrated[:, i] = y_b

        # ROC-AUC
        if len(np.unique(y_t)) > 1:
            roc_auc = float(roc_auc_score(y_t, y_s))
            macro_roc.append(roc_auc)
        else:
            roc_auc = 0.50

        # PR-AUC
        if len(np.unique(y_t)) > 1 and y_t.sum() > 0:
            pr_auc = float(average_precision_score(y_t, y_s))
            macro_pr.append(pr_auc)
        else:
            pr_auc = float(y_t.mean()) if len(y_t) > 0 else 0.0

        # Confusion Matrix
        cm = confusion_matrix(y_t, y_b, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / len(y_t) if len(y_t) > 0 else 0.0

        ece = compute_ece(y_t, y_s)
        brier = float(brier_score_loss(y_t, y_s))

        macro_f1.append(f1)
        macro_sens.append(sens)
        macro_spec.append(spec)
        macro_prec.append(prec)
        macro_ece.append(ece)
        macro_brier.append(brier)

        results_clf[disease] = {
            "Optimal_Threshold": tau,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
            "Accuracy": acc,
            "F1": f1,
            "ECE": ece,
            "Brier_Score": brier,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Positives": int(y_t.sum()),
        }

    # Binary Normal vs Abnormal
    true_abn = (y_true.sum(axis=1) > 0).astype(int)
    pred_abn = (y_bin_calibrated.sum(axis=1) > 0).astype(int)
    bin_cm = confusion_matrix(true_abn, pred_abn, labels=[0, 1])
    bin_tn, bin_fp, bin_fn, bin_tp = int(bin_cm[0, 0]), int(bin_cm[0, 1]), int(bin_cm[1, 0]), int(bin_cm[1, 1])

    # 4. Weakly Supervised Detection
    print("\n[Evaluating Weakly Supervised Localization on Ground Truth Bounding Boxes...]")
    bbox_df = load_bounding_boxes(BBOX_FILE)
    gradcam = GradCAM(model)

    ious_all = []
    det_samples = 0
    max_bbox = 50

    for _, row in bbox_df.iterrows():
        if det_samples >= max_bbox:
            break
        img_filename = row["Image_Index"]
        disease_label = row["Finding_Label"].strip().lower().replace(" ", "_")
        if disease_label == "infiltrate":
            disease_label = "infiltration"

        img_path = IMAGE_DIR / img_filename
        if not img_path.exists():
            continue

        matching_indices = [
            i for i, d in enumerate(DISEASE_CLASSES)
            if d.lower() == disease_label or d.replace("_", " ").lower() == disease_label
        ]
        if not matching_indices:
            continue
        idx = matching_indices[0]

        pil_img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = pil_img.size
        tensor = preprocess_image_for_inference(pil_img).to(device)

        cam = gradcam.generate(tensor, target_class_idx=idx, target_size=(w_orig, h_orig))
        pred_boxes = extract_bounding_boxes(cam, threshold=0.45)
        gt_box = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))

        best_iou = max([compute_iou(pb, gt_box) for pb in pred_boxes]) if pred_boxes else 0.0
        ious_all.append(best_iou)
        det_samples += 1

    ious_all = np.array(ious_all) if ious_all else np.array([0.0])
    mean_iou = float(np.mean(ious_all))

    # mAP@0.5 and mAP@0.5:0.95
    iou_steps = np.arange(0.50, 1.00, 0.05)
    prec_at_steps = [float(np.mean(ious_all >= t)) for t in iou_steps]
    map_50 = prec_at_steps[0]
    map_50_95 = float(np.mean(prec_at_steps))

    detection_results = {
        "Evaluated_BBoxes": len(ious_all),
        "Mean_IoU": mean_iou,
        "mAP@0.5": map_50,
        "mAP@0.5:0.95": map_50_95,
        "Detection_Precision@0.5": map_50,
        "Detection_Recall@0.5": map_50,
        "Detection_Precision@0.25": float(np.mean(ious_all >= 0.25)),
        "Detection_Recall@0.25": float(np.mean(ious_all >= 0.25)),
        "Detection_Rate@0.10": float(np.mean(ious_all >= 0.10)),
    }

    full_report = {
        "classification": {
            "per_class": results_clf,
            "macro_metrics": {
                "Macro_ROC_AUC": float(np.mean(macro_roc)),
                "Macro_PR_AUC": float(np.mean(macro_pr)),
                "Macro_Sensitivity": float(np.mean(macro_sens)),
                "Macro_Specificity": float(np.mean(macro_spec)),
                "Macro_Precision": float(np.mean(macro_prec)),
                "Macro_F1": float(np.mean(macro_f1)),
                "Expected_Calibration_Error": float(np.mean(macro_ece)),
                "Mean_Brier_Score": float(np.mean(macro_brier)),
            },
            "binary_normal_abnormal": {
                "Accuracy": float(accuracy_score(true_abn, pred_abn)),
                "Precision": float(precision_score(true_abn, pred_abn, zero_division=0)),
                "Sensitivity": float(recall_score(true_abn, pred_abn, zero_division=0)),
                "Specificity": float(bin_tn / (bin_tn + bin_fp)) if (bin_tn + bin_fp) > 0 else 1.0,
                "Confusion_Matrix": {"TN": bin_tn, "FP": bin_fp, "FN": bin_fn, "TP": bin_tp},
            },
        },
        "detection": detection_results,
    }

    report_path = OUTPUT_DIR / "final_calibrated_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print("\n--- FINAL EVALUATION COMPLETE ---")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
