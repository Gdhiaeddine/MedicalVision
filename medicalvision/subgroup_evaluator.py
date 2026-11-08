"""
MedicalVision AI - Subgroup Fairness and Robustness Evaluator.
Evaluates model performance across clinical and demographic subgroups:
1. Patient Sex (Male vs Female)
2. Patient Age (<40 young, 40-65 adult, >65 senior)
3. Radiographic View Position (PA - Posteroanterior vs AP - Anteroposterior)
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

from medicalvision.config import DISEASE_CLASSES


def evaluate_cohort(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float],
    disease_classes: List[str] = DISEASE_CLASSES
) -> Dict[str, Any]:
    """
    Computes AUROC, Sensitivity, Specificity, F1, TP, FP, TN, FN for a subgroup cohort.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return {"n_samples": 0}
        
    per_class = {}
    valid_aurocs = []
    sensitivities = []
    specificities = []
    f1s = []

    for i, disease in enumerate(disease_classes):
        target = y_true[:, i]
        prob = y_prob[:, i]
        thresh = thresholds.get(disease, 0.50)
        preds = (prob >= thresh).astype(int)

        n_pos = int(np.sum(target))
        n_neg = int(len(target) - n_pos)

        # ROC-AUC (only valid if both classes exist)
        if len(np.unique(target)) > 1:
            auc = float(roc_auc_score(target, prob))
            valid_aurocs.append(auc)
        else:
            auc = None

        if n_pos > 0 and n_neg > 0:
            tn, fp, fn, tp = confusion_matrix(target, preds, labels=[0, 1]).ravel()
        elif n_pos > 0 and n_neg == 0:
            tp = int(np.sum(preds == 1))
            fn = int(np.sum(preds == 0))
            fp, tn = 0, 0
        else:
            tn = int(np.sum(preds == 0))
            fp = int(np.sum(preds == 1))
            tp, fn = 0, 0

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

        if n_pos > 0:
            sensitivities.append(sens)
            f1s.append(f1)
        if n_neg > 0:
            specificities.append(spec)

        per_class[disease] = {
            "n_positives": n_pos,
            "roc_auc": round(auc, 4) if auc is not None else None,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    macro_auc = round(float(np.mean(valid_aurocs)), 4) if len(valid_aurocs) > 0 else None
    macro_sens = round(float(np.mean(sensitivities)), 4) if len(sensitivities) > 0 else 0.0
    macro_spec = round(float(np.mean(specificities)), 4) if len(specificities) > 0 else 0.0
    macro_f1 = round(float(np.mean(f1s)), 4) if len(f1s) > 0 else 0.0

    return {
        "n_samples": n_samples,
        "macro_roc_auc": macro_auc,
        "macro_sensitivity": macro_sens,
        "macro_specificity": macro_spec,
        "macro_f1": macro_f1,
        "evaluated_classes_auc_count": len(valid_aurocs),
        "per_class": per_class,
    }


def run_subgroup_analysis(
    metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float],
    disease_classes: List[str] = DISEASE_CLASSES
) -> Dict[str, Any]:
    """
    Splits cohort by Sex, Age, and View Position and computes full comparative statistics.
    """
    report = {
        "sex_subgroups": {},
        "age_subgroups": {},
        "view_subgroups": {},
        "clinical_bias_alerts": []
    }

    # 1. Patient Sex: M vs F
    for sex in ["M", "F"]:
        mask = (metadata_df["Patient Sex"] == sex).to_numpy()
        cohort_name = "Male" if sex == "M" else "Female"
        report["sex_subgroups"][cohort_name] = evaluate_cohort(
            y_true[mask], y_prob[mask], thresholds, disease_classes
        )

    # Check for Sex Disparity (> 0.08 difference in Macro AUC)
    auc_m = report["sex_subgroups"]["Male"].get("macro_roc_auc")
    auc_f = report["sex_subgroups"]["Female"].get("macro_roc_auc")
    if auc_m is not None and auc_f is not None and abs(auc_m - auc_f) > 0.08:
        report["clinical_bias_alerts"].append(
            f"Significant gender disparity observed: Macro ROC-AUC Male={auc_m:.4f} vs Female={auc_f:.4f} (diff > 0.08)."
        )

    # 2. Patient Age: <40, 40-65, >65
    age_groups = {
        "Young (<40)": metadata_df["Patient Age"] < 40,
        "Adult (40-65)": (metadata_df["Patient Age"] >= 40) & (metadata_df["Patient Age"] <= 65),
        "Senior (>65)": metadata_df["Patient Age"] > 65,
    }
    for group_name, condition in age_groups.items():
        mask = condition.to_numpy()
        report["age_subgroups"][group_name] = evaluate_cohort(
            y_true[mask], y_prob[mask], thresholds, disease_classes
        )

    # 3. View Position: PA vs AP
    for view in ["PA", "AP"]:
        mask = (metadata_df["View Position"] == view).to_numpy()
        view_label = f"View_{view} ({'Posteroanterior' if view == 'PA' else 'Anteroposterior'})"
        report["view_subgroups"][view_label] = evaluate_cohort(
            y_true[mask], y_prob[mask], thresholds, disease_classes
        )

    # Check AP vs PA Cardiomegaly sensitivity / false positive disparity
    pa_cardio = report["view_subgroups"].get("View_PA (Posteroanterior)", {}).get("per_class", {}).get("Cardiomegaly", {})
    ap_cardio = report["view_subgroups"].get("View_AP (Anteroposterior)", {}).get("per_class", {}).get("Cardiomegaly", {})
    if pa_cardio and ap_cardio:
        spec_pa = pa_cardio.get("specificity", 1.0)
        spec_ap = ap_cardio.get("specificity", 1.0)
        if spec_pa - spec_ap > 0.15:
            report["clinical_bias_alerts"].append(
                f"Cardiomegaly magnification artifact confirmed: AP view specificity ({spec_ap:.2f}) is markedly lower than PA view ({spec_pa:.2f})."
            )

    return report
