"""
MedicalVision AI - Evaluate Faster R-CNN on NIH BBox Test Set.
Computes mAP@0.5, mAP@0.5:0.95, Precision, Recall, and per-disease metrics.
"""

import json
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader

from medicalvision.config import CHECKPOINT_DIR, OUTPUT_DIR, DETECTION_CLASSES
from medicalvision.detection_dataset import (
    get_detection_split,
    detection_collate_fn,
    IDX_TO_CLASS,
)
from medicalvision.models.detector import CXRDetector
from medicalvision.detection_metrics import evaluate_detection_predictions


def evaluate_detector(
    checkpoint_path: Path = CHECKPOINT_DIR / "faster_rcnn_nih.pth",
    output_json: Path = OUTPUT_DIR / "detector_evaluation_report.json",
    confidence_threshold: float = 0.01,
) -> Dict:
    print("=" * 60)
    print(" MedicalVision AI — Faster R-CNN Object Detection Evaluation")
    print("=" * 60)

    train_dataset, test_dataset, _ = get_detection_split(train_ratio=0.8, random_seed=42)
    print(f"Evaluating on {len(test_dataset)} independent test images (patient-level split)...")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=0,
    )

    detector = CXRDetector(
        checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
        confidence_threshold=confidence_threshold,
        device=torch.device("cpu"),
    )

    all_gt = {}
    all_preds = {}
    total_imgs = len(test_loader)

    for idx, (images, targets) in enumerate(test_loader):
        if (idx + 1) % 20 == 0 or idx == total_imgs - 1:
            print(f"  Processed [{idx + 1}/{total_imgs}] test images...")
        img_tensor = images[0]
        target = targets[0]
        img_name = target["image_name"]

        # Parse Ground Truth
        gt_boxes = target["boxes"].numpy()
        gt_labels = target["labels"].numpy()

        all_gt[img_name] = {}
        for b, l in zip(gt_boxes, gt_labels):
            class_name = IDX_TO_CLASS.get(int(l))
            if class_name:
                all_gt[img_name].setdefault(class_name, []).append(b)

        # Run Predictions
        preds = detector.predict(img_tensor)
        all_preds[img_name] = {}
        for p in preds:
            c = p["label"]
            score = p["score"]
            box = np.array(p["box"])
            all_preds[img_name].setdefault(c, []).append((box, score))

    metrics = evaluate_detection_predictions(all_gt, all_preds, DETECTION_CLASSES)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f" Object Detection Evaluation Complete")
    print(f" mAP@0.5:      {metrics['mAP_at_50'] * 100:.2f} %")
    print(f" mAP@0.5:0.95: {metrics['mAP_at_50_95'] * 100:.2f} %")
    print("=" * 60)
    print(f"{'Pathology':<18} | {'AP@0.5':<8} | {'AP@COCO':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 60)
    for c, vals in metrics["per_class"].items():
        print(f"{c:<18} | {vals['ap_at_50']*100:>6.2f}% | {vals['ap_coco_50_95']*100:>6.2f}% | {vals['precision_at_50']*100:>8.2f}% | {vals['recall_at_50']*100:>6.2f}%")
    print("=" * 60)
    print(f"Full detection report saved to {output_json}")

    return metrics


if __name__ == "__main__":
    evaluate_detector()
