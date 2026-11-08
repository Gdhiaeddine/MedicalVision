"""
MedicalVision AI - Patient-Level Paired Bootstrap Engine.
Computes 95% Confidence Intervals (2.5th and 97.5th percentiles) and paired model differences
for clinical screening and classification metrics.
Guarantees patient-level clustering (all images belonging to a sampled patient are included).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    f1_score,
)


def evaluate_binary_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Computes standard screening metrics given continuous probabilities and a frozen threshold."""
    preds = (probs >= threshold).astype(int)
    positives = int(np.sum(y_true))
    negatives = int(len(y_true) - positives)

    if positives == 0 or negatives == 0:
        return {
            "sensitivity": np.nan,
            "specificity": 1.0 if negatives > 0 else np.nan,
            "precision": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "tp": 0,
            "fp": 0,
            "tn": negatives,
            "fn": positives,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    f2 = float(5 * prec * sens / (4 * prec + sens)) if (4 * prec + sens) > 0 else 0.0

    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else np.nan
    prauc = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else np.nan

    return {
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "f2": f2,
        "roc_auc": auc,
        "pr_auc": prauc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def paired_patient_bootstrap(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    prob_14: np.ndarray,
    prob_5: np.ndarray,
    threshold_14: float,
    threshold_5: float,
    repetitions: int = 3000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs paired patient-level bootstrap resampling.
    1. Resamples unique patient IDs with replacement.
    2. Gathers all images belonging to the resampled patient IDs.
    3. Evaluates both models simultaneously.
    4. Calculates point estimates, 95% CIs, and paired differences (M5 - M14).
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    # Index mapping: patient_id -> array of image row indices
    patient_to_indices = {}
    for idx, pid in enumerate(patient_ids):
        patient_to_indices.setdefault(pid, []).append(idx)

    metrics_list = ["sensitivity", "specificity", "precision", "f1", "f2", "roc_auc", "pr_auc"]
    dist_14 = {m: [] for m in metrics_list}
    dist_5 = {m: [] for m in metrics_list}
    differences = {m: [] for m in metrics_list}

    # Baseline point estimates on full test set
    base_14 = evaluate_binary_predictions(y_true, prob_14, threshold_14)
    base_5 = evaluate_binary_predictions(y_true, prob_5, threshold_5)

    valid_reps = 0
    for _ in range(repetitions):
        # Sample patient IDs with replacement
        sampled_pts = rng.choice(unique_patients, size=n_patients, replace=True)
        # Collect all image indices
        sampled_indices = []
        for p in sampled_pts:
            sampled_indices.extend(patient_to_indices[p])

        sampled_indices = np.array(sampled_indices)
        y_b = y_true[sampled_indices]

        # Guard: sample must contain both positive and negative cases
        if len(np.unique(y_b)) < 2:
            continue

        p14_b = prob_14[sampled_indices]
        p5_b = prob_5[sampled_indices]

        m14 = evaluate_binary_predictions(y_b, p14_b, threshold_14)
        m5 = evaluate_binary_predictions(y_b, p5_b, threshold_5)

        for m in metrics_list:
            v14 = m14[m]
            v5 = m5[m]
            if not np.isnan(v14) and not np.isnan(v5):
                dist_14[m].append(v14)
                dist_5[m].append(v5)
                differences[m].append(v5 - v14)

        valid_reps += 1

    summary = {
        "repetitions_requested": repetitions,
        "valid_repetitions_completed": valid_reps,
        "metrics": {},
    }

    for m in metrics_list:
        diff_arr = np.array(differences[m])
        arr_14 = np.array(dist_14[m])
        arr_5 = np.array(dist_5[m])

        diff_pt = base_5[m] - base_14[m]
        diff_ci_lower = float(np.percentile(diff_arr, 2.5))
        diff_ci_upper = float(np.percentile(diff_arr, 97.5))
        is_stat_sig = bool((diff_ci_lower > 0) or (diff_ci_upper < 0))

        summary["metrics"][m] = {
            "model_14": {
                "point_estimate": round(float(base_14[m]), 4),
                "ci_95": [round(float(np.percentile(arr_14, 2.5)), 4), round(float(np.percentile(arr_14, 97.5)), 4)],
            },
            "model_5": {
                "point_estimate": round(float(base_5[m]), 4),
                "ci_95": [round(float(np.percentile(arr_5, 2.5)), 4), round(float(np.percentile(arr_5, 97.5)), 4)],
            },
            "difference": {
                "point_estimate": round(float(diff_pt), 4),
                "ci_95": [round(diff_ci_lower, 4), round(diff_ci_upper, 4)],
                "statistically_significant": is_stat_sig,
            },
        }

    return summary
