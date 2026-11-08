"""
MedicalVision AI - Binary Screener Module.
Provides calibrated Normal vs Abnormal global screening to eliminate false positive explosion
inherent to accumulating 14 independent disease thresholds.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix


class BinaryScreener:
    """
    Two-stage screening decision engine.
    Stage 1 determines if an image is Normal or Abnormal with high specificity.
    Stage 2 (if Abnormal) runs differential multi-label classification.
    """

    def __init__(self, target_specificity: float = 0.82):
        self.target_specificity = target_specificity
        self.meta_model: Optional[LogisticRegression] = None
        self.optimal_threshold: float = 0.50
        self.joint_prob_threshold: float = 0.50

    def fit_meta_model(self, logits: np.ndarray, y_binary: np.ndarray):
        """
        Fit a regularized logistic regression model on the 14 raw logits to predict
        whether at least one abnormality is present.
        """
        self.meta_model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=500)
        self.meta_model.fit(logits, y_binary)
        
        # Optimize threshold on probabilities for target specificity
        probs = self.meta_model.predict_proba(logits)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_binary, probs)
        
        # Specificity = 1 - FPR. Find threshold where Specificity >= target_specificity
        valid_indices = np.where(1.0 - fpr >= self.target_specificity)[0]
        if len(valid_indices) > 0:
            # Pick the threshold in valid indices that maximizes sensitivity (TPR)
            best_idx = valid_indices[np.argmax(tpr[valid_indices])]
            self.optimal_threshold = float(thresholds[best_idx])
        else:
            # Fallback to Youden's J
            j_scores = tpr - fpr
            self.optimal_threshold = float(thresholds[np.argmax(j_scores)])

    def fit_joint_prob_screener(self, calib_probs: np.ndarray, y_binary: np.ndarray):
        """
        Calibrate screening using joint independence:
        P(Abnormal) = 1 - Prod(1 - p_i)
        """
        # Clip to prevent log/numerical overflow
        p_clipped = np.clip(calib_probs, 1e-6, 1.0 - 1e-6)
        # Probability of No Finding = product of (1 - p_i)
        p_normal = np.prod(1.0 - p_clipped, axis=1)
        joint_abnormal = 1.0 - p_normal
        
        fpr, tpr, thresholds = roc_curve(y_binary, joint_abnormal)
        valid_indices = np.where(1.0 - fpr >= self.target_specificity)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(tpr[valid_indices])]
            self.joint_prob_threshold = float(thresholds[best_idx])
        else:
            j_scores = tpr - fpr
            self.joint_prob_threshold = float(thresholds[np.argmax(j_scores)])

    def predict_abnormal(self, logits: Optional[np.ndarray] = None, calib_probs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predictions_binary, probabilities_abnormal).
        """
        if self.meta_model is not None and logits is not None:
            probs = self.meta_model.predict_proba(logits)[:, 1]
            preds = (probs >= self.optimal_threshold).astype(int)
            return preds, probs
        elif calib_probs is not None:
            p_clipped = np.clip(calib_probs, 1e-6, 1.0 - 1e-6)
            p_normal = np.prod(1.0 - p_clipped, axis=1)
            probs = 1.0 - p_normal
            preds = (probs >= self.joint_prob_threshold).astype(int)
            return preds, probs
        else:
            raise ValueError("Either logits or calib_probs must be provided.")

    def evaluate(self, y_true_binary: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        """Compute standard clinical screening evaluation metrics."""
        auc = float(roc_auc_score(y_true_binary, probs)) if len(np.unique(y_true_binary)) > 1 else 0.5
        tn, fp, fn, tp = confusion_matrix(y_true_binary, preds).ravel()
        
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0 # Precision
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0 # Negative Predictive Value
        f1 = float(2 * ppv * sensitivity / (ppv + sensitivity)) if (ppv + sensitivity) > 0 else 0.0
        
        return {
            "roc_auc": round(auc, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision_ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "f1_score": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }
