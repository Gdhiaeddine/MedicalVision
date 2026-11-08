"""
MedicalVision AI - Comprehensive Head-to-Head Comparison:
Dedicated 5-Class Model vs Current 14-Class Model.
Evaluates:
1. Per-Class & Macro ROC-AUC
2. Per-Class & Macro PR-AUC (Average Precision)
3. Recall / Sensitivity & Specificity
4. Macro F1 and F2-Score
5. Binary Meta-Screener Performance (False Positives, Specificity, Sensitivity, PPV)
6. Probability Calibration (Brier Skill Score & ECE)
"""

import os
import json
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss

from medicalvision.config import DISEASE_CLASSES, CHECKPOINT_DIR, OUTPUT_DIR
from medicalvision.config_5class import FIVE_DISEASE_CLASSES, CHECKPOINT_5CLASS, COMPARISON_REPORT
from medicalvision.dataset_5class import get_5class_patient_splits, FiveClassChestXrayDataset
from medicalvision.transforms import get_eval_transforms
from medicalvision.models.classifier import build_model
from medicalvision.binary_screener import BinaryScreener
from medicalvision.calibration import PlattScaler, compute_calibration_report, compute_ece, plot_reliability_diagrams
from medicalvision.thresholding import find_optimal_thresholds, compute_metrics_at_threshold


def evaluate_model_on_5classes(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts logits, targets, and raw probabilities."""
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    logits_cat = torch.cat(all_logits, dim=0).numpy()
    targets_cat = torch.cat(all_targets, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_cat))
    return logits_cat, targets_cat, probs


def run_comparison(
    n_val: int = 200,
    n_test: int = 400,
):
    print("=" * 70)
    print(" Head-to-Head Comparison: Specialized 5-Class vs Current 14-Class")
    print(" Target 5 Conditions: " + ", ".join(FIVE_DISEASE_CLASSES))
    print("=" * 70)

    device = torch.device("cpu")
    _, val_df, test_df = get_5class_patient_splits()

    val_sub = val_df.sample(n=min(n_val, len(val_df)), random_state=42).reset_index(drop=True)
    test_sub = test_df.sample(n=min(n_test, len(test_df)), random_state=42).reset_index(drop=True)

    val_ds = FiveClassChestXrayDataset(val_sub, transform=get_eval_transforms())
    test_ds = FiveClassChestXrayDataset(test_sub, transform=get_eval_transforms())

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"Validation cohort: {len(val_ds)} scans | Test cohort: {len(test_ds)} scans")

    # Map the 5 class indices in the 14-class model
    idx_14_to_5 = [DISEASE_CLASSES.index(c) for c in FIVE_DISEASE_CLASSES]

    # --- Model A: Current 14-Class Model (restricting outputs to the 5 classes) ---
    print("\n[1/3] Evaluating Model A: Current 14-Class DenseNet-121...")
    ckpt_14 = CHECKPOINT_DIR / "best_model.pth"
    model_14 = build_model("densenet121", num_classes=14, pretrained=False, checkpoint_path=str(ckpt_14) if ckpt_14.exists() else None)
    model_14.to(device).eval()

    val_logits_14_raw, val_targets_5, _ = evaluate_model_on_5classes(model_14, val_loader, device)
    test_logits_14_raw, test_targets_5, _ = evaluate_model_on_5classes(model_14, test_loader, device)

    val_logits_14 = val_logits_14_raw[:, idx_14_to_5]
    test_logits_14 = test_logits_14_raw[:, idx_14_to_5]

    # --- Model B: Dedicated 5-Class Model ---
    print("\n[2/3] Evaluating Model B: Dedicated 5-Class DenseNet-121...")
    model_5 = build_model("densenet121", num_classes=len(FIVE_DISEASE_CLASSES), pretrained=False)
    checkpoint_5 = torch.load(CHECKPOINT_5CLASS, map_location=device, weights_only=False)
    model_5.load_state_dict(checkpoint_5["model_state_dict"])
    model_5.to(device).eval()

    val_logits_5, _, _ = evaluate_model_on_5classes(model_5, val_loader, device)
    test_logits_5, _, _ = evaluate_model_on_5classes(model_5, test_loader, device)

    # --- Calibration & Thresholds & Screening for Both ---
    print("\n[3/3] Calibrating & Benchmarking Decision Systems...")

    def evaluate_pipeline(name: str, val_logits: np.ndarray, test_logits: np.ndarray):
        # 1. Platt Scaler
        platt = PlattScaler(num_classes=5)
        platt.fit(val_logits, val_targets_5)
        val_probs = platt.transform(val_logits)
        test_probs = platt.transform(test_logits)

        # 2. Meta-Screener (target specificity 80%)
        val_binary = (val_targets_5.sum(axis=1) > 0).astype(int)
        test_binary = (test_targets_5.sum(axis=1) > 0).astype(int)

        screener = BinaryScreener(target_specificity=0.80)
        screener.fit_meta_model(val_logits, val_binary)
        preds_screen, probs_screen = screener.predict_abnormal(logits=test_logits)
        screen_eval = screener.evaluate(test_binary, preds_screen, probs_screen)

        # 3. Constrained Multi-Label Thresholds (Spec >= 75%)
        per_disease = {}
        aurocs = []
        praucs = []
        recalls = []
        specificities = []
        f1s = []
        f2s = []
        eces = []

        for c, disease in enumerate(FIVE_DISEASE_CLASSES):
            yt_val = val_targets_5[:, c]
            yp_val = val_probs[:, c]
            yt_test = test_targets_5[:, c]
            yp_test = test_probs[:, c]

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

            aurocs.append(auc)
            praucs.append(prauc)
            recalls.append(m["recall"])
            specificities.append(m["specificity"])
            f1s.append(m["f1"])
            f2s.append(m["f2"])
            eces.append(ece)

            per_disease[disease] = {
                "threshold": round(th, 4),
                "roc_auc": round(auc, 4),
                "pr_auc": round(prauc, 4),
                "recall": m["recall"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "f2": m["f2"],
                "ece": round(ece, 4),
                "brier_score": round(brier, 4),
                "brier_reference": round(b_ref, 4),
                "brier_skill_score": round(bss, 4),
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }

        return {
            "name": name,
            "macro_roc_auc": round(float(np.mean(aurocs)), 4),
            "macro_pr_auc": round(float(np.mean(praucs)), 4),
            "macro_recall": round(float(np.mean(recalls)), 4),
            "macro_specificity": round(float(np.mean(specificities)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "macro_f2": round(float(np.mean(f2s)), 4),
            "macro_ece": round(float(np.mean(eces)), 4),
            "macro_brier_score": round(float(np.mean([per_disease[d]["brier_score"] for d in FIVE_DISEASE_CLASSES])), 4),
            "macro_brier_skill_score": round(float(np.mean([per_disease[d]["brier_skill_score"] for d in FIVE_DISEASE_CLASSES])), 4),
            "screening_metrics": screen_eval,
            "per_disease": per_disease,
        }

    eval_14 = evaluate_pipeline("14-Class Model (Restricted to Top 5)", val_logits_14, test_logits_14)
    eval_5 = evaluate_pipeline("Dedicated 5-Class Model", val_logits_5, test_logits_5)

    comparison_bundle = {
        "model_14class_restricted": eval_14,
        "model_5class_dedicated": eval_5,
        "verdict": {
            "roc_auc_delta": round(eval_5["macro_roc_auc"] - eval_14["macro_roc_auc"], 4),
            "pr_auc_delta": round(eval_5["macro_pr_auc"] - eval_14["macro_pr_auc"], 4),
            "f1_delta": round(eval_5["macro_f1"] - eval_14["macro_f1"], 4),
            "screening_fp_delta": eval_5["screening_metrics"]["false_positives"] - eval_14["screening_metrics"]["false_positives"],
            "screening_specificity_delta": round(eval_5["screening_metrics"]["specificity"] - eval_14["screening_metrics"]["specificity"], 4),
        }
    }

    COMPARISON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_REPORT, "w") as f:
        json.dump(comparison_bundle, f, indent=2)

    print("\n" + "=" * 70)
    print(" HEAD-TO-HEAD COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'14-Class Model (Top 5)':<22} | {'Dedicated 5-Class Model':<22}")
    print("-" * 75)
    print(f"{'Macro ROC-AUC':<25} | {eval_14['macro_roc_auc']:>22.4f} | {eval_5['macro_roc_auc']:>22.4f}")
    print(f"{'Macro PR-AUC':<25} | {eval_14['macro_pr_auc']:>22.4f} | {eval_5['macro_pr_auc']:>22.4f}")
    print(f"{'Macro Recall':<25} | {eval_14['macro_recall']*100:>21.2f}% | {eval_5['macro_recall']*100:>21.2f}%")
    print(f"{'Macro Specificity':<25} | {eval_14['macro_specificity']*100:>21.2f}% | {eval_5['macro_specificity']*100:>21.2f}%")
    print(f"{'Macro F1-Score':<25} | {eval_14['macro_f1']:>22.4f} | {eval_5['macro_f1']:>22.4f}")
    print(f"{'Macro F2-Score':<25} | {eval_14['macro_f2']:>22.4f} | {eval_5['macro_f2']:>22.4f}")
    print(f"{'Macro ECE (Calibration)':<25} | {eval_14['macro_ece']*100:>21.2f}% | {eval_5['macro_ece']*100:>21.2f}%")
    print("-" * 75)
    print(f"{'Screening False Positives':<25} | {eval_14['screening_metrics']['false_positives']:>22} | {eval_5['screening_metrics']['false_positives']:>22}")
    print(f"{'Screening Specificity':<25} | {eval_14['screening_metrics']['specificity']*100:>21.2f}% | {eval_5['screening_metrics']['specificity']*100:>21.2f}%")
    print(f"{'Screening Sensitivity':<25} | {eval_14['screening_metrics']['sensitivity']*100:>21.2f}% | {eval_5['screening_metrics']['sensitivity']*100:>21.2f}%")
    print(f"{'Screening Precision (PPV)':<25} | {eval_14['screening_metrics']['precision_ppv']*100:>21.2f}% | {eval_5['screening_metrics']['precision_ppv']*100:>21.2f}%")
    print("=" * 70)
    print(f"Report saved to {COMPARISON_REPORT}")

    return comparison_bundle


if __name__ == "__main__":
    run_comparison()
