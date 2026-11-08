"""
MedicalVision AI - Advanced Multi-Label Threshold Optimization Benchmark.
Compares:
1. Fixed standard threshold (0.50)
2. Youden's Index (TPR - FPR)
3. F1-Score Maximization
4. F2-Score Maximization (Recall-Prioritized)
5. Constrained Clinical Threshold (Max Sensitivity subject to Specificity >= 75%)
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def compute_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute precision, recall, specificity, F1, F2, FP, FN at given threshold."""
    preds = (y_score >= threshold).astype(int)
    positives = int(np.sum(y_true))
    negatives = int(len(y_true) - positives)
    
    if positives == 0 or negatives == 0:
        return {
            "threshold": round(float(threshold), 4),
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 1.0 if negatives > 0 else 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": negatives,
            "fn": positives
        }
        
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    f2 = float(5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0.0
    
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def find_optimal_thresholds(y_val_true: np.ndarray, y_val_probs: np.ndarray) -> Dict[str, float]:
    """
    Find optimal threshold for each strategy on validation data for a single class.
    Strategies:
    - 'fixed_05': 0.50
    - 'youden': max(TPR - FPR)
    - 'f1': max F1
    - 'f2': max F2
    - 'constrained_spec75': max Recall with Specificity >= 0.75
    """
    if len(np.unique(y_val_true)) < 2:
        return {
            "fixed_05": 0.50,
            "youden": 0.50,
            "f1": 0.50,
            "f2": 0.50,
            "constrained_spec75": 0.50
        }
        
    thresholds = np.linspace(0.01, 0.99, 1000)
    
    best_youden_val = -float("inf")
    best_youden_thresh = 0.50
    
    best_f1_val = -float("inf")
    best_f1_thresh = 0.50
    
    best_f2_val = -float("inf")
    best_f2_thresh = 0.50
    
    best_constrained_recall = -float("inf")
    best_constrained_thresh = 0.50
    
    # Vectorized computation for candidate thresholds
    y_true_bool = y_val_true.astype(bool)
    n_pos = np.sum(y_true_bool)
    n_neg = len(y_val_true) - n_pos
    
    for t in thresholds:
        preds = y_val_probs >= t
        tp = np.sum(preds & y_true_bool)
        fp = np.sum(preds & (~y_true_bool))
        fn = n_pos - tp
        tn = n_neg - fp
        
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        spec = tn / n_neg if n_neg > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Youden
        j = tpr - fpr
        if j > best_youden_val:
            best_youden_val = j
            best_youden_thresh = t
            
        # F1
        f1 = (2 * prec * tpr) / (prec + tpr) if (prec + tpr) > 0 else 0.0
        if f1 > best_f1_val:
            best_f1_val = f1
            best_f1_thresh = t
            
        # F2
        f2 = (5 * prec * tpr) / (4 * prec + tpr) if (4 * prec + tpr) > 0 else 0.0
        if f2 > best_f2_val:
            best_f2_val = f2
            best_f2_thresh = t
            
        # Constrained: Specificity >= 0.75, maximize Recall (TPR)
        if spec >= 0.75:
            if tpr > best_constrained_recall:
                best_constrained_recall = tpr
                best_constrained_thresh = t
                
    return {
        "fixed_05": 0.50,
        "youden": float(best_youden_thresh),
        "f1": float(best_f1_thresh),
        "f2": float(best_f2_thresh),
        "constrained_spec75": float(best_constrained_thresh)
    }


def benchmark_all_strategies(
    y_val_true: np.ndarray,
    y_val_probs: np.ndarray,
    y_test_true: np.ndarray,
    y_test_probs: np.ndarray,
    disease_names: List[str]
) -> Dict[str, Dict]:
    """
    Fits thresholds on validation set, then evaluates all strategies on test set.
    """
    strategies = ["fixed_05", "youden", "f1", "f2", "constrained_spec75"]
    results = {strat: {"per_class": {}, "macro": {}} for strat in strategies}
    
    num_classes = y_val_true.shape[1]
    
    # Store chosen thresholds
    chosen_thresholds = {strat: {} for strat in strategies}
    
    for c in range(num_classes):
        disease = disease_names[c]
        val_t = y_val_true[:, c]
        val_p = y_val_probs[:, c]
        test_t = y_test_true[:, c]
        test_p = y_test_probs[:, c]
        
        optimal_t = find_optimal_thresholds(val_t, val_p)
        
        for strat in strategies:
            th = optimal_t[strat]
            chosen_thresholds[strat][disease] = th
            eval_metrics = compute_metrics_at_threshold(test_t, test_p, th)
            results[strat]["per_class"][disease] = eval_metrics
            
    # Macro averages per strategy
    for strat in strategies:
        classes_with_positives = [
            d for d in disease_names 
            if results[strat]["per_class"][d]["tp"] + results[strat]["per_class"][d]["fn"] > 0
        ]
        
        macro_prec = float(np.mean([results[strat]["per_class"][d]["precision"] for d in classes_with_positives]))
        macro_rec = float(np.mean([results[strat]["per_class"][d]["recall"] for d in classes_with_positives]))
        macro_spec = float(np.mean([results[strat]["per_class"][d]["specificity"] for d in disease_names]))
        macro_f1 = float(np.mean([results[strat]["per_class"][d]["f1"] for d in classes_with_positives]))
        macro_f2 = float(np.mean([results[strat]["per_class"][d]["f2"] for d in classes_with_positives]))
        
        total_tp = int(sum([results[strat]["per_class"][d]["tp"] for d in disease_names]))
        total_fp = int(sum([results[strat]["per_class"][d]["fp"] for d in disease_names]))
        total_fn = int(sum([results[strat]["per_class"][d]["fn"] for d in disease_names]))
        total_tn = int(sum([results[strat]["per_class"][d]["tn"] for d in disease_names]))
        
        results[strat]["macro"] = {
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_specificity": round(macro_spec, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_f2": round(macro_f2, 4),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_tn": total_tn,
        }
        results[strat]["thresholds"] = chosen_thresholds[strat]
        
    return results
