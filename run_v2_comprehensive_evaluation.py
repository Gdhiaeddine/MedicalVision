"""
MedicalVision AI - Version 2.0 Comprehensive Evaluation Master Script.
Executes the 5 major clinical & technical improvements:
1. Binary Screening Optimization (re-evaluates normal vs abnormal to slash false positives).
2. Multi-Label Threshold Benchmark (Youden vs F1 vs F2 vs Constrained Spec>=75%).
3. Calibration Audit: Brier Skill Score (BSS), ECE per condition, and Reliability Diagrams.
4. Subgroup Analysis: Sex (M vs F), Age (<40, 40-65, >65), View Position (PA vs AP).
5. Aggregates all metrics into structured clinical JSON and Markdown artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from medicalvision.config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    METADATA_FILE,
    DISEASE_CLASSES,
)
from medicalvision.dataset import (
    load_dataset_metadata,
    split_by_patient,
    ChestXrayDataset,
    get_eval_transforms,
)
from medicalvision.models.classifier import build_model
from medicalvision.calibration import (
    TemperatureScaler,
    PlattScaler,
    compute_calibration_report,
    plot_reliability_diagrams,
)
from medicalvision.binary_screener import BinaryScreener
from medicalvision.thresholding import benchmark_all_strategies
from medicalvision.subgroup_evaluator import run_subgroup_analysis


def run_full_v2_evaluation(
    model_checkpoint: str = "outputs/checkpoints/best_model.pth",
    n_val_patients: int = 250,
    n_test_patients: int = 500,
    batch_size: int = 16,
):
    print("=" * 70)
    print(" MedicalVision AI (v2.0) — Clinical Verification & Fairness Audit")
    print("=" * 70)

    device = torch.device("cpu")

    # 1. Load Datasets with zero-leakage patient-level split
    print("\n[Step 1/5] Loading Patient-Level Datasets...")
    full_df = load_dataset_metadata(METADATA_FILE)
    train_df, val_df, test_df = split_by_patient(full_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    # Subsample validation and test to fixed patient size for fast evaluation
    rng = np.random.RandomState(42)
    val_sub_df = val_df.sample(n=min(n_val_patients, len(val_df)), random_state=42).reset_index(drop=True)
    test_sub_df = test_df.sample(n=min(n_test_patients, len(test_df)), random_state=42).reset_index(drop=True)

    eval_transforms = get_eval_transforms()
    val_dataset = ChestXrayDataset(val_sub_df, transform=eval_transforms)
    test_dataset = ChestXrayDataset(test_sub_df, transform=eval_transforms)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Validation cohort: {len(val_dataset)} scans | Test cohort: {len(test_dataset)} scans")

    # 2. Load Model
    print("\n[Step 2/5] Loading Fine-Tuned DenseNet-121 Checkpoint...")
    model = build_model("densenet121", num_classes=len(DISEASE_CLASSES), pretrained=False)
    if os.path.exists(model_checkpoint):
        checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded weights from {model_checkpoint}")
    else:
        print(f"Warning: {model_checkpoint} not found! Using initialized model.")

    model.to(device)
    model.eval()

    # Collect logits and targets
    def extract_logits_and_targets(loader):
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for imgs, targets, _ in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
        return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)

    print("Extracting validation and test logits...")
    val_logits_t, val_targets_t = extract_logits_and_targets(val_loader)
    test_logits_t, test_targets_t = extract_logits_and_targets(test_loader)

    val_targets = val_targets_t.numpy()
    test_targets = test_targets_t.numpy()

    # 3. Fit Temperature Scaler & Class-wise Platt Scaler
    print("\n[Step 3/5] Probability Calibration (Temperature Scaling & Platt Scaling)...")
    scaler = TemperatureScaler()
    best_temp = scaler.fit(val_logits_t, val_targets_t)
    print(f"Learned Global Temperature T: {best_temp:.4f}")

    platt_scaler = PlattScaler(num_classes=len(DISEASE_CLASSES))
    platt_scaler.fit(val_logits_t.detach().numpy(), val_targets)
    print("Fitted Class-Specific Platt Scalers for all 14 pathologies.")

    with torch.no_grad():
        val_uncal_probs = torch.sigmoid(val_logits_t).detach().numpy()
        test_uncal_probs = torch.sigmoid(test_logits_t).detach().numpy()

    # Use Platt-scaled probabilities for true clinical calibration and positive BSS
    val_calib_probs = platt_scaler.transform(val_logits_t.detach().numpy())
    test_calib_probs = platt_scaler.transform(test_logits_t.detach().numpy())

    # Calibration Report (BSS & ECE per disease)
    calib_report = compute_calibration_report(
        test_targets,
        test_uncal_probs,
        test_calib_probs,
        DISEASE_CLASSES,
    )
    # Generate Reliability Diagrams
    diagram_path = plot_reliability_diagrams(
        test_targets,
        test_calib_probs,
        output_path="outputs/reliability_diagrams_14_classes.png",
        disease_classes=DISEASE_CLASSES,
    )
    print(f"Generated 14 Reliability Diagrams: {diagram_path}")

    # 4. Binary Screening Optimization
    print("\n[Step 4/5] Binary Screening Optimization (False Positive Reduction)...")
    val_binary = (val_targets.sum(axis=1) > 0).astype(int)
    test_binary = (test_targets.sum(axis=1) > 0).astype(int)

    # Target specificity 82%
    screener = BinaryScreener(target_specificity=0.82)
    screener.fit_meta_model(val_logits_t.detach().numpy(), val_binary)
    screener.fit_joint_prob_screener(val_calib_probs, val_binary)

    # Evaluate meta-screener on test set
    meta_preds, meta_probs = screener.predict_abnormal(logits=test_logits_t.detach().numpy())
    screener_metrics = screener.evaluate(test_binary, meta_preds, meta_probs)

    # Evaluate baseline rule: "Abnormal if ANY disease exceeds fixed 0.50 or Youden"
    naive_fixed_preds = (test_uncal_probs >= 0.50).any(axis=1).astype(int)
    naive_fixed_metrics = screener.evaluate(test_binary, naive_fixed_preds, naive_fixed_preds.astype(float))

    print(f"  Naive Threshold Screening : {naive_fixed_metrics['false_positives']} False Positives | Spec: {naive_fixed_metrics['specificity']*100:.1f}%")
    print(f"  Calibrated Meta-Screener  : {screener_metrics['false_positives']} False Positives | Spec: {screener_metrics['specificity']*100:.1f}% | Sens: {screener_metrics['sensitivity']*100:.1f}%")
    fp_reduction = naive_fixed_metrics['false_positives'] - screener_metrics['false_positives']
    print(f"  --> False Positives Eliminated: {fp_reduction} cases ({fp_reduction / max(1, naive_fixed_metrics['false_positives']) * 100:.1f}% reduction)")

    # 5. Multi-Label Threshold Benchmark
    print("\n[Step 5/5] Multi-Label Threshold Benchmark (Fixed vs Youden vs F1 vs F2 vs Constrained)...")
    threshold_benchmark = benchmark_all_strategies(
        y_val_true=val_targets,
        y_val_probs=val_calib_probs,
        y_test_true=test_targets,
        y_test_probs=test_calib_probs,
        disease_names=DISEASE_CLASSES,
    )

    print("\nThreshold Benchmark Comparison (Macro on Test Cohort):")
    print(f"{'Strategy':<20} | {'Macro Prec':<12} | {'Macro Rec':<12} | {'Macro F1':<10} | {'Macro F2':<10} | {'Total FP':<8} | {'Total FN':<8}")
    print("-" * 90)
    for strat, data in threshold_benchmark.items():
        m = data["macro"]
        print(f"{strat:<20} | {m['macro_precision']*100:>10.2f}% | {m['macro_recall']*100:>10.2f}% | {m['macro_f1']:>10.4f} | {m['macro_f2']:>10.4f} | {m['total_fp']:>8} | {m['total_fn']:>8}")

    # 6. Subgroup Fairness & Disparity Analysis
    print("\n[Audit] Subgroup Fairness Evaluation (Sex, Age, View Position)...")
    test_metadata = test_sub_df

    # Use F2 thresholds for subgroup evaluation
    f2_thresholds = threshold_benchmark["f2"]["thresholds"]
    subgroup_report = run_subgroup_analysis(
        metadata_df=test_metadata,
        y_true=test_targets,
        y_prob=test_calib_probs,
        thresholds=f2_thresholds,
        disease_classes=DISEASE_CLASSES,
    )

    # Export full comprehensive results
    output_bundle = {
        "binary_screener": {
            "meta_screener_optimal_threshold": screener.optimal_threshold,
            "screener_performance": screener_metrics,
            "naive_fixed_baseline": naive_fixed_metrics,
            "fp_reduction_count": fp_reduction,
        },
        "threshold_benchmark": threshold_benchmark,
        "calibration_audit": {
            "optimal_temperature": best_temp,
            "per_disease": calib_report,
            "macro_brier_skill_score": float(np.mean([calib_report[d]["Brier_Skill_Score"] for d in DISEASE_CLASSES])),
            "macro_ece_calibrated": float(np.mean([calib_report[d]["ECE_Calibrated"] for d in DISEASE_CLASSES])),
            "macro_ece_uncalibrated": float(np.mean([calib_report[d]["ECE_Uncalibrated"] for d in DISEASE_CLASSES])),
        },
        "subgroup_fairness": subgroup_report,
    }

    result_path = OUTPUT_DIR / "v2_comprehensive_evaluation_report.json"
    with open(result_path, "w") as f:
        json.dump(output_bundle, f, indent=2)

    print(f"\n Master v2.0 evaluation report saved to {result_path}")
    return output_bundle


if __name__ == "__main__":
    run_full_v2_evaluation()
