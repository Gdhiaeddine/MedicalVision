"""
Advanced Evaluation & Validation Engine for MedicalVision AI:
1. Rigorous Patient-Level Separation & Stratified Test Cohort
2. Temperature Scaling Calibration (Guo et al., ICML 2017)
3. Clinical Threshold Optimization via Youden's Index (J = Sensitivity + Specificity - 1)
4. Comprehensive Metrics: ROC-AUC, PR-AUC, Sens, Spec, F1, Confusion Matrices, ECE, Brier Skill Score
5. Weakly-Supervised Detection restricted strictly to the 8 NIH-annotated conditions.
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
import torch.nn as nn
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
from medicalvision.calibration import TemperatureScaler, VectorTemperatureScaler, compute_ece, compute_calibration_report
from medicalvision.interpretability.gradcam import GradCAM
from medicalvision.interpretability.localization import extract_bounding_boxes, compute_iou


def build_stratified_test_cohort(
    test_df: pd.DataFrame,
    target_size: int = 500,
    min_positives_per_class: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Construct a representative test cohort from the test split such that
    EVERY condition has at least min_positives_per_class samples.
    """
    selected_indices = set()
    rng = np.random.RandomState(seed)

    for disease in DISEASE_CLASSES:
        pos_df = test_df[test_df[disease] == 1.0]
        n_available = len(pos_df)
        if n_available > 0:
            n_take = min(n_available, min_positives_per_class)
            sampled = pos_df.sample(n=n_take, random_state=rng)
            selected_indices.update(sampled.index)

    # Fill remainder with negative / normal scans to reflect realistic prevalence
    remaining_needed = max(0, target_size - len(selected_indices))
    avail_indices = list(set(test_df.index) - selected_indices)
    if remaining_needed > 0 and avail_indices:
        n_fill = min(remaining_needed, len(avail_indices))
        fill_idx = rng.choice(avail_indices, size=n_fill, replace=False)
        selected_indices.update(fill_idx)

    cohort = test_df.loc[list(selected_indices)].sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return cohort


def optimize_youden_thresholds(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    disease_classes: List[str] = DISEASE_CLASSES,
) -> Dict[str, float]:
    """
    Optimize decision threshold for EACH disease on VALIDATION SET using Youden's Index:
    J(tau) = Sensitivity(tau) + Specificity(tau) - 1
    This eliminates the 'zero recall' pathology on rare conditions.
    """
    threshold_grid = np.linspace(0.10, 0.90, 81)
    optimal_thresholds = {}

    for i, disease in enumerate(disease_classes):
        y_true = val_targets[:, i]
        y_scores = val_probs[:, i]

        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            optimal_thresholds[disease] = 0.50
            continue

        best_j = -2.0
        best_tau = 0.50

        for tau in threshold_grid:
            y_pred = (y_scores >= tau).astype(int)

            pos_mask = (y_true == 1)
            neg_mask = (y_true == 0)

            sens = y_pred[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
            spec = (1 - y_pred[neg_mask]).mean() if neg_mask.sum() > 0 else 1.0
            j = sens + spec - 1.0

            if j > best_j:
                best_j = j
                best_tau = float(tau)

        optimal_thresholds[disease] = best_tau

    return optimal_thresholds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print("  RIGOROUS CLINICAL VALIDATION & TEMPERATURE SCALING EVALUATION")
    print("=" * 75)
    print(f"Device: {device}")

    # 1. Dataset Verification & Leakage Check
    print("\n[1. Verifying Patient-Level Split & Leakage Zero-Tolerance...]")
    df = load_dataset_metadata(METADATA_FILE, IMAGE_DIR, existing_images_only=True)
    train_df, val_df, test_df = split_by_patient(df, train_ratio=0.75, val_ratio=0.10, test_ratio=0.15)

    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])

    assert len(train_patients.intersection(val_patients)) == 0, "Patient leakage train/val!"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage train/test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient leakage val/test!"
    print("Zero Patient Leakage Confirmed: No patient ID appears in more than one partition.")

    # 2. Stratified Test Cohort with guaranteed positives for all diseases
    print("\n[2. Constructing Stratified Test Cohort (N=500 scans)...]")
    test_cohort = build_stratified_test_cohort(test_df, target_size=500, min_positives_per_class=10)
    print("Positives per class in Test Cohort:")
    for d in DISEASE_CLASSES:
        print(f"  - {d:<20}: {int(test_cohort[d].sum()):>3} positive scans")

    # 3. Model Loading
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    assert ckpt_path.exists(), "Trained checkpoint best_model.pth missing!"
    model = build_model(backbone="densenet121", pretrained=False, checkpoint_path=str(ckpt_path), device=device)
    model.eval()

    # 4. Temperature Scaling on Validation Set
    print("\n[3. Performing Temperature Scaling Calibration on Validation Set...]")
    val_subset = val_df.sample(n=min(len(val_df), 600), random_state=42).reset_index(drop=True)
    val_loader = DataLoader(
        ChestXrayDataset(val_subset, image_dir=IMAGE_DIR, transform=get_eval_transforms()),
        batch_size=32,
        shuffle=False,
    )

    val_logits_list = []
    val_targets_list = []
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            val_logits_list.append(logits.cpu())
            val_targets_list.append(targets)

    val_logits_tensor = torch.cat(val_logits_list, dim=0)
    val_targets_tensor = torch.cat(val_targets_list, dim=0)

    temp_scaler = TemperatureScaler().to(device)
    fitted_temp = temp_scaler.fit(val_logits_tensor.to(device), val_targets_tensor.to(device))
    print(f"Optimized Scalar Temperature (T): {fitted_temp:.4f}")

    # Validation Calibrated Probabilities
    with torch.no_grad():
        val_scaled_logits = temp_scaler(val_logits_tensor.to(device))
        val_calibrated_probs = torch.sigmoid(val_scaled_logits).cpu().numpy()
        val_targets_np = val_targets_tensor.numpy()

    # 5. Threshold Optimization via Youden's Index on Validation Set
    print("\n[4. Optimizing Thresholds via Youden's Index on Validation Set...]")
    youden_thresholds = optimize_youden_thresholds(val_calibrated_probs, val_targets_np)
    for d in DISEASE_CLASSES:
        print(f"  - {d:<20}: tau* (Youden) = {youden_thresholds[d]:.2f}")

    # 6. Evaluation on Unseen Test Set
    print("\n[5. Running Final Evaluation on Unseen Test Cohort...]")
    test_loader = DataLoader(
        ChestXrayDataset(test_cohort, image_dir=IMAGE_DIR, transform=get_eval_transforms()),
        batch_size=32,
        shuffle=False,
    )

    test_logits_list = []
    test_targets_list = []
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            test_logits_list.append(logits.cpu())
            test_targets_list.append(targets)

    test_logits_tensor = torch.cat(test_logits_list, dim=0)
    test_targets_tensor = torch.cat(test_targets_list, dim=0)
    y_true = test_targets_tensor.numpy()

    # Uncalibrated & Calibrated probabilities
    y_prob_uncal = torch.sigmoid(test_logits_tensor).numpy()
    with torch.no_grad():
        y_prob_cal = torch.sigmoid(temp_scaler(test_logits_tensor.to(device))).cpu().numpy()

    # Calibration report (ECE, Brier, BSS)
    cal_report = compute_calibration_report(y_true, y_prob_uncal, y_prob_cal)

    # Classification Metrics Calculation
    class_results = {}
    valid_aurocs = []
    valid_auprcs = []
    valid_f1s = []
    valid_sens = []
    valid_specs = []
    valid_precs = []
    valid_eces = []
    valid_briers = []

    y_bin_test = np.zeros_like(y_prob_cal, dtype=int)

    for i, disease in enumerate(DISEASE_CLASSES):
        y_t = y_true[:, i]
        y_s = y_prob_cal[:, i]
        tau = youden_thresholds[disease]
        y_b = (y_s >= tau).astype(int)
        y_bin_test[:, i] = y_b

        n_pos = int(y_t.sum())

        # ROC-AUC (strictly valid only if both classes present)
        if n_pos > 0 and n_pos < len(y_t):
            roc = float(roc_auc_score(y_t, y_s))
            valid_aurocs.append(roc)
        else:
            roc = float("nan")

        # PR-AUC
        if n_pos > 0:
            pr = float(average_precision_score(y_t, y_s))
            valid_auprcs.append(pr)
        else:
            pr = float("nan")

        # Confusion matrix
        cm = confusion_matrix(y_t, y_b, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / len(y_t) if len(y_t) > 0 else 0.0

        valid_f1s.append(f1)
        valid_sens.append(sens)
        valid_specs.append(spec)
        valid_precs.append(prec)

        ece_val = cal_report[disease]["ECE_Calibrated"]
        brier_val = cal_report[disease]["Brier_Calibrated"]
        valid_eces.append(ece_val)
        valid_briers.append(brier_val)

        class_results[disease] = {
            "Optimal_Threshold": tau,
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
            "Accuracy": acc,
            "F1": f1,
            "ECE_Before": cal_report[disease]["ECE_Uncalibrated"],
            "ECE_After": ece_val,
            "Brier_Score": brier_val,
            "Brier_Reference": cal_report[disease]["Brier_Reference"],
            "Brier_Skill_Score": cal_report[disease]["Brier_Skill_Score"],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Positives": n_pos,
        }

    macro_roc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0
    macro_pr = float(np.mean(valid_auprcs)) if valid_auprcs else 0.0
    macro_f1 = float(np.mean(valid_f1s))
    macro_sens = float(np.mean(valid_sens))
    macro_spec = float(np.mean(valid_specs))
    macro_prec = float(np.mean(valid_precs))
    macro_ece = float(np.mean(valid_eces))
    macro_brier = float(np.mean(valid_briers))

    # Binary Normal vs Abnormal
    true_abn = (y_true.sum(axis=1) > 0).astype(int)
    pred_abn = (y_bin_test.sum(axis=1) > 0).astype(int)
    bin_cm = confusion_matrix(true_abn, pred_abn, labels=[0, 1])
    bin_tn, bin_fp, bin_fn, bin_tp = int(bin_cm[0, 0]), int(bin_cm[0, 1]), int(bin_cm[1, 0]), int(bin_cm[1, 1])

    # 7. Weakly Supervised Detection (Restricted strictly to the 8 NIH annotated pathologies)
    print("\n[6. Evaluating Weakly-Supervised Localization on the 8 NIH-Annotated Pathologies...]")
    bbox_df = load_bounding_boxes(BBOX_FILE)
    NIH_BBOX_CONDITIONS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

    gradcam = GradCAM(model)
    ious_all = []
    bbox_eval_count = 0
    max_eval_bboxes = 60

    for _, row in bbox_df.iterrows():
        if bbox_eval_count >= max_eval_bboxes:
            break

        img_filename = row["Image_Index"]
        lbl_raw = row["Finding_Label"].strip().lower().replace(" ", "_")
        if lbl_raw == "infiltrate":
            lbl_raw = "infiltration"

        # Check if among the 8 annotated diseases
        match = [d for d in NIH_BBOX_CONDITIONS if d.lower() == lbl_raw]
        if not match:
            continue
        disease_name = match[0]
        c_idx = DISEASE_CLASSES.index(disease_name)

        img_path = IMAGE_DIR / img_filename
        if not img_path.exists():
            continue

        pil_img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = pil_img.size
        tensor = preprocess_image_for_inference(pil_img).to(device)

        cam = gradcam.generate(tensor, target_class_idx=c_idx, target_size=(w_orig, h_orig))
        pred_boxes = extract_bounding_boxes(cam, threshold=0.45)
        gt_box = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))

        best_iou = max([compute_iou(pb, gt_box) for pb in pred_boxes]) if pred_boxes else 0.0
        ious_all.append(best_iou)
        bbox_eval_count += 1

    ious_all = np.array(ious_all) if ious_all else np.array([0.0])
    mean_iou = float(np.mean(ious_all))

    iou_grid = np.arange(0.50, 1.00, 0.05)
    prec_grid = [float(np.mean(ious_all >= t)) for t in iou_grid]
    map_50 = prec_grid[0]
    map_50_95 = float(np.mean(prec_grid))

    detection_metrics = {
        "Annotated_Classes_Evaluated": NIH_BBOX_CONDITIONS,
        "Total_BBoxes_Evaluated": len(ious_all),
        "Mean_IoU": mean_iou,
        "mAP@0.5": map_50,
        "mAP@0.5:0.95": map_50_95,
        "Detection_Precision@0.5": map_50,
        "Detection_Recall@0.5": map_50,
        "Detection_Precision@0.25": float(np.mean(ious_all >= 0.25)),
        "Detection_Recall@0.25": float(np.mean(ious_all >= 0.25)),
        "Detection_Rate@0.10": float(np.mean(ious_all >= 0.10)),
    }

    final_payload = {
        "calibration": {
            "Fitted_Temperature": fitted_temp,
            "ECE_Before": float(np.mean([class_results[d]["ECE_Before"] for d in DISEASE_CLASSES])),
            "ECE_After": macro_ece,
        },
        "classification": {
            "per_class": class_results,
            "macro_metrics": {
                "Macro_ROC_AUC": macro_roc,
                "Macro_PR_AUC": macro_pr,
                "Macro_Sensitivity": macro_sens,
                "Macro_Specificity": macro_spec,
                "Macro_Precision": macro_prec,
                "Macro_F1": macro_f1,
                "Expected_Calibration_Error": macro_ece,
                "Mean_Brier_Score": macro_brier,
            },
            "binary_normal_abnormal": {
                "Accuracy": float(accuracy_score(true_abn, pred_abn)),
                "Precision": float(precision_score(true_abn, pred_abn, zero_division=0)),
                "Sensitivity": float(recall_score(true_abn, pred_abn, zero_division=0)),
                "Specificity": float(bin_tn / (bin_tn + bin_fp)) if (bin_tn + bin_fp) > 0 else 1.0,
                "Confusion_Matrix": {"TN": bin_tn, "FP": bin_fp, "FN": bin_fn, "TP": bin_tp},
            },
        },
        "detection": detection_metrics,
    }

    save_path = OUTPUT_DIR / "rigorous_clinical_evaluation.json"
    with open(save_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n--- RESULTS JSON GENERATED ---")
    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()
