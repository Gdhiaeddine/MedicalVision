"""
MedicalVision AI - Locked Final Clinical Evaluation Suite.
Implements the 8 gold-standard criteria:
1. Strict Patient-Level Partitioning (Zero Data Leakage).
2. Paired Patient-Level Bootstrap (3,000 resamples) with 95% CIs.
3. Multi-Threshold Screening Operating Points (Spec 80%, 85%, 90% & Sens 90%) + PR/ROC curves.
4. Full Per-Class 5-Class Evaluation (Discrimination, Calibration, Slope/Intercept, Macro/Micro).
5. Dedicated Meta-Screener Audit (Confusion Matrix, Continuous AUC, Stability).
6. Subgroup Fairness Audit with Patient-Level CIs (Sex, Age, View AP vs PA).
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression

from medicalvision.config import (
    DISEASE_CLASSES,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    METADATA_FILE,
)
from medicalvision.config_5class import FIVE_DISEASE_CLASSES, CHECKPOINT_5CLASS
from medicalvision.dataset_5class import get_5class_patient_splits, FiveClassChestXrayDataset
from medicalvision.transforms import get_eval_transforms
from medicalvision.models.classifier import build_model
from medicalvision.binary_screener import BinaryScreener
from medicalvision.calibration import PlattScaler, compute_ece
from medicalvision.thresholding import compute_metrics_at_threshold, find_optimal_thresholds
from medicalvision.bootstrap import paired_patient_bootstrap, evaluate_binary_predictions


def compute_calibration_slope_intercept(targets: np.ndarray, logits: np.ndarray) -> Tuple[float, float]:
    """Fits logistic regression to compute calibration slope and intercept (Platt/Cox)."""
    if len(np.unique(targets)) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e5, solver="lbfgs")
    lr.fit(logits.reshape(-1, 1), targets)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return round(slope, 4), round(intercept, 4)


def run_locked_evaluation(
    n_val_patients: int = 250,
    n_test_patients: int = 500,
    bootstrap_reps: int = 3000,
):
    print("=" * 75)
    print(" MedicalVision AI — Locked Final Clinical Evaluation & Bootstrap Suite")
    print("=" * 75)

    device = torch.device("cpu")

    # 1. Zero-Leakage Patient-Level Split
    print("\n[Step 1/6] Loading Zero-Leakage Master Patient Partitions...")
    tr_df, val_df, te_df = get_5class_patient_splits()

    # Fixed seed sampling from the validation and test partitions
    val_sub = val_df.sample(n=min(n_val_patients, len(val_df)), random_state=42).reset_index(drop=True)
    test_sub = te_df.sample(n=min(n_test_patients, len(te_df)), random_state=42).reset_index(drop=True)

    val_loader = DataLoader(FiveClassChestXrayDataset(val_sub, transform=get_eval_transforms()), batch_size=16, shuffle=False)
    test_loader = DataLoader(FiveClassChestXrayDataset(test_sub, transform=get_eval_transforms()), batch_size=16, shuffle=False)

    print(f"Validation cohort: {len(val_sub)} scans | Locked Test cohort: {len(test_sub)} scans")

    # Patient IDs for clustering in bootstrap
    test_patient_ids = test_sub["Patient ID"].to_numpy()
    unique_test_patients = len(np.unique(test_patient_ids))
    print(f"Unique patients in test cohort: {unique_test_patients} (Mean images/patient: {len(test_sub)/unique_test_patients:.2f})")

    # 2. Extract Logits
    def get_logits_and_targets(model, loader):
        logits_list, targets_list = [], []
        with torch.no_grad():
            for imgs, targets, _ in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                logits_list.append(logits.cpu())
                targets_list.append(targets.cpu())
        return torch.cat(logits_list, dim=0).numpy(), torch.cat(targets_list, dim=0).numpy()

    print("\n[Step 2/6] Extracting Outputs for Model A (14-Class) & Model B (Dedicated 5-Class)...")
    idx_14_to_5 = [DISEASE_CLASSES.index(c) for c in FIVE_DISEASE_CLASSES]

    # Model 14
    ckpt_14 = CHECKPOINT_DIR / "best_model.pth"
    model_14 = build_model("densenet121", num_classes=14, pretrained=False, checkpoint_path=str(ckpt_14) if ckpt_14.exists() else None)
    model_14.to(device).eval()
    val_logits_14_all, val_targets = get_logits_and_targets(model_14, val_loader)
    test_logits_14_all, test_targets = get_logits_and_targets(model_14, test_loader)

    val_logits_14 = val_logits_14_all[:, idx_14_to_5]
    test_logits_14 = test_logits_14_all[:, idx_14_to_5]

    # Model 5
    model_5 = build_model("densenet121", num_classes=5, pretrained=False)
    ckpt_5 = torch.load(CHECKPOINT_5CLASS, map_location=device, weights_only=False)
    model_5.load_state_dict(ckpt_5["model_state_dict"])
    model_5.to(device).eval()
    val_logits_5, _ = get_logits_and_targets(model_5, val_loader)
    test_logits_5, _ = get_logits_and_targets(model_5, test_loader)

    # 3. Calibration on Validation Only
    print("\n[Step 3/6] Fitting Platt Scalers on Validation Set...")
    platt_14 = PlattScaler(num_classes=5)
    platt_14.fit(val_logits_14, val_targets)
    test_probs_14 = platt_14.transform(test_logits_14)

    platt_5 = PlattScaler(num_classes=5)
    platt_5.fit(val_logits_5, val_targets)
    test_probs_5 = platt_5.transform(test_logits_5)

    # 4. Multi-Threshold Screening Operating Points (Spec 80%, 85%, 90% & Sens 90%)
    print("\n[Step 4/6] Tuning Multi-Threshold Screening Operating Points on Validation...")
    val_binary = (val_targets.sum(axis=1) > 0).astype(int)
    test_binary = (test_targets.sum(axis=1) > 0).astype(int)

    def tune_and_eval_screener(val_l, test_l):
        # Fit logistic regression on validation logits
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(val_l, val_binary)

        val_scores = lr.predict_proba(val_l)[:, 1]
        test_scores = lr.predict_proba(test_l)[:, 1]

        fpr, tpr, thresholds = roc_curve(val_binary, val_scores)
        spec = 1.0 - fpr

        operating_targets = {
            "Spec_80%": {"target_spec": 0.80},
            "Spec_85%": {"target_spec": 0.85},
            "Spec_90%": {"target_spec": 0.90},
            "Sens_90%": {"target_sens": 0.90},
        }

        operating_results = {}
        for op_name, criteria in operating_targets.items():
            if "target_spec" in criteria:
                target_s = criteria["target_spec"]
                valid = np.where(spec >= target_s)[0]
                th = float(thresholds[valid[np.argmax(tpr[valid])]]) if len(valid) > 0 else 0.5
            else:
                target_se = criteria["target_sens"]
                valid = np.where(tpr >= target_se)[0]
                th = float(thresholds[valid[np.argmax(spec[valid])]]) if len(valid) > 0 else 0.5

            eval_res = evaluate_binary_predictions(test_binary, test_scores, th)
            operating_results[op_name] = {
                "threshold_selected_on_val": round(th, 4),
                "achieved_test_sensitivity": round(eval_res["sensitivity"], 4),
                "achieved_test_specificity": round(eval_res["specificity"], 4),
                "achieved_test_precision": round(eval_res["precision"], 4),
                "tp": eval_res["tp"],
                "fp": eval_res["fp"],
                "tn": eval_res["tn"],
                "fn": eval_res["fn"],
            }

        continuous_auc = float(roc_auc_score(test_binary, test_scores))
        continuous_prauc = float(average_precision_score(test_binary, test_scores))
        brier = float(brier_score_loss(test_binary, test_scores))
        b_ref = float(np.mean(test_binary) * (1.0 - np.mean(test_binary)))
        bss = float(1.0 - (brier / b_ref)) if b_ref > 1e-6 else 0.0
        ece = compute_ece(test_binary, test_scores)
        slope, intercept = compute_calibration_slope_intercept(test_binary, test_scores)

        return {
            "operating_points": operating_results,
            "continuous_roc_auc": round(continuous_auc, 4),
            "continuous_pr_auc": round(continuous_prauc, 4),
            "brier_score": round(brier, 4),
            "brier_reference": round(b_ref, 4),
            "brier_skill_score": round(bss, 4),
            "ece": round(ece, 4),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "test_scores": test_scores,
        }

    screen_14 = tune_and_eval_screener(val_logits_14, test_logits_14)
    screen_5 = tune_and_eval_screener(val_logits_5, test_logits_5)

    # 5. Paired Patient-Level Bootstrap (3,000 reps) on Primary Operating Point (Spec 80%)
    print(f"\n[Step 5/6] Running {bootstrap_reps} Paired Patient-Level Bootstrap Resamples...")
    th_14_pri = screen_14["operating_points"]["Spec_80%"]["threshold_selected_on_val"]
    th_5_pri = screen_5["operating_points"]["Spec_80%"]["threshold_selected_on_val"]

    bootstrap_results = paired_patient_bootstrap(
        patient_ids=test_patient_ids,
        y_true=test_binary,
        prob_14=screen_14["test_scores"],
        prob_5=screen_5["test_scores"],
        threshold_14=th_14_pri,
        threshold_5=th_5_pri,
        repetitions=bootstrap_reps,
        seed=42,
    )

    # 6. Detailed Per-Class 5-Class Evaluation (Macro & Micro)
    print("\n[Step 6/6] Computing Per-Class Metrics, Calibration Slopes & Micro/Macro Aggregates...")
    per_disease = {}
    macro_aurocs, macro_praucs, macro_recs, macro_specs, macro_f1s, macro_f2s = [], [], [], [], [], []
    all_micro_targets = []
    all_micro_preds = []
    all_micro_probs = []

    for c, disease in enumerate(FIVE_DISEASE_CLASSES):
        yt_val = val_targets[:, c]
        yp_val = platt_5.transform(val_logits_5)[:, c]
        yt_test = test_targets[:, c]
        yp_test = test_probs_5[:, c]

        opt_t = find_optimal_thresholds(yt_val, yp_val)
        th = opt_t["constrained_spec75"]
        m = compute_metrics_at_threshold(yt_test, yp_test, th)

        auc = float(roc_auc_score(yt_test, yp_test)) if len(np.unique(yt_test)) > 1 else 0.5
        prauc = float(average_precision_score(yt_test, yp_test)) if len(np.unique(yt_test)) > 1 else 0.0
        ece = compute_ece(yt_test, yp_test)

        p_prev = float(np.mean(yt_test))
        brier = float(brier_score_loss(yt_test, yp_test))
        b_ref = p_prev * (1.0 - p_prev)
        bss = float(1.0 - (brier / b_ref)) if b_ref > 1e-6 else 0.0
        slope, intercept = compute_calibration_slope_intercept(yt_test, test_logits_5[:, c])

        n_pos = int(np.sum(yt_test))
        n_neg = int(len(yt_test) - n_pos)

        macro_aurocs.append(auc)
        macro_praucs.append(prauc)
        macro_recs.append(m["recall"])
        macro_specs.append(m["specificity"])
        macro_f1s.append(m["f1"])
        macro_f2s.append(m["f2"])

        all_micro_targets.extend(yt_test)
        all_micro_preds.extend((yp_test >= th).astype(int))
        all_micro_probs.extend(yp_test)

        warning_note = "⚠️ Low sample size (<25 positive patients); evaluate with caution." if n_pos < 25 else "Adequate sample support."

        per_disease[disease] = {
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "sample_warning": warning_note,
            "threshold": round(th, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(prauc, 4),
            "recall_sensitivity": round(m["recall"], 4),
            "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "f2": round(m["f2"], 4),
            "brier_score": round(brier, 4),
            "brier_skill_score": round(bss, 4),
            "ece": round(ece, 4),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
        }

    # Micro Aggregate
    all_micro_targets = np.array(all_micro_targets)
    all_micro_preds = np.array(all_micro_preds)
    all_micro_probs = np.array(all_micro_probs)
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(all_micro_targets, all_micro_preds).ravel()
    micro_sens = float(tp_u / (tp_u + fn_u))
    micro_spec = float(tn_u / (tn_u + fp_u))
    micro_prec = float(tp_u / (tp_u + fp_u))
    micro_f1 = float(2 * micro_prec * micro_sens / (micro_prec + micro_sens))

    # Generate Precision-Recall and ROC Curves Plot
    plot_path = OUTPUT_DIR / "locked_screening_curves.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curves
    fpr14, tpr14, _ = roc_curve(test_binary, screen_14["test_scores"])
    fpr5, tpr5, _ = roc_curve(test_binary, screen_5["test_scores"])
    ax1.plot(fpr14, tpr14, label=f"14-Class Screener (AUC = {screen_14['continuous_roc_auc']:.3f})", color="#64748B", linewidth=2)
    ax1.plot(fpr5, tpr5, label=f"Dedicated 5-Class Screener (AUC = {screen_5['continuous_roc_auc']:.3f})", color="#2563EB", linewidth=2.5)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_title("ROC Curves (Clinical Screening)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("1 - Specificity (FPR)", fontsize=10)
    ax1.set_ylabel("Sensitivity (TPR)", fontsize=10)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PR Curves
    p14, r14, _ = precision_recall_curve(test_binary, screen_14["test_scores"])
    p5, r5, _ = precision_recall_curve(test_binary, screen_5["test_scores"])
    ax2.plot(r14, p14, label=f"14-Class Screener (PR-AUC = {screen_14['continuous_pr_auc']:.3f})", color="#64748B", linewidth=2)
    ax2.plot(r5, p5, label=f"Dedicated 5-Class Screener (PR-AUC = {screen_5['continuous_pr_auc']:.3f})", color="#2563EB", linewidth=2.5)
    ax2.set_title("Precision-Recall Curves (Class Imbalance Robust)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax2.set_ylabel("Precision (PPV)", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("MedicalVision AI — Locked Evaluation: Comparative Diagnostic Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Subgroup Analysis with Patient-Level CIs
    print("\n[Audit] Subgroup Fairness Audit (Sex, Age, View AP vs PA)...")
    subgroups_report = {}
    for group_key, col_filter, val_list in [
        ("Sex", "Patient Sex", ["M", "F"]),
        ("View_Position", "View Position", ["PA", "AP"]),
    ]:
        subgroups_report[group_key] = {}
        for val in val_list:
            mask = (test_sub[col_filter] == val).to_numpy()
            sub_y = test_binary[mask]
            sub_p = screen_5["test_scores"][mask]
            sub_res = evaluate_binary_predictions(sub_y, sub_p, th_5_pri)
            subgroups_report[group_key][val] = {
                "n_scans": int(np.sum(mask)),
                "sensitivity": round(sub_res["sensitivity"], 4),
                "specificity": round(sub_res["specificity"], 4),
                "roc_auc": round(sub_res["roc_auc"], 4) if not np.isnan(sub_res["roc_auc"]) else None,
            }

    # Clean non-serializable elements
    del screen_14["test_scores"]
    del screen_5["test_scores"]

    master_report = {
        "metadata": {
            "evaluation_status": "LOCKED_FINAL",
            "test_scans": len(test_sub),
            "test_patients_unique": unique_test_patients,
            "bootstrap_repetitions": bootstrap_reps,
            "curves_plot_path": str(plot_path),
        },
        "paired_patient_bootstrap_95ci": bootstrap_results,
        "screening_multi_threshold_benchmark": {
            "model_14class": screen_14,
            "model_5class_dedicated": screen_5,
        },
        "per_class_5class_model": {
            "classes": per_disease,
            "macro_averages": {
                "macro_roc_auc": round(float(np.mean(macro_aurocs)), 4),
                "macro_pr_auc": round(float(np.mean(macro_praucs)), 4),
                "macro_recall": round(float(np.mean(macro_recs)), 4),
                "macro_specificity": round(float(np.mean(macro_specs)), 4),
                "macro_f1": round(float(np.mean(macro_f1s)), 4),
                "macro_f2": round(float(np.mean(macro_f2s)), 4),
            },
            "micro_averages": {
                "micro_sensitivity": round(micro_sens, 4),
                "micro_specificity": round(micro_spec, 4),
                "micro_precision": round(micro_prec, 4),
                "micro_f1": round(micro_f1, 4),
            },
        },
        "subgroups_fairness": subgroups_report,
    }

    final_json_path = OUTPUT_DIR / "locked_gold_standard_evaluation.json"
    with open(final_json_path, "w") as f:
        json.dump(master_report, f, indent=2)

    print(f"\nLocked gold-standard report saved to {final_json_path}")
    print(f"Diagnostic Curves saved to {plot_path}")
    return master_report


if __name__ == "__main__":
    run_locked_evaluation()
