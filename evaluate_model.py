"""
Standalone evaluation & benchmark script for MedicalVision AI.
Calculates AUROC, F1, Sensitivity, Specificity, and Weakly-Supervised Bounding Box IoU.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch

from medicalvision.config import (
    DISEASE_CLASSES,
    IMAGE_DIR,
    METADATA_FILE,
    BBOX_FILE,
    CHECKPOINT_DIR,
    DEFAULT_THRESHOLD,
)
from medicalvision.dataset import load_dataset_metadata, split_by_patient, load_bounding_boxes, ChestXrayDataset
from medicalvision.models.classifier import build_model
from medicalvision.evaluator import evaluate_dataset_predictions, evaluate_localization_iou
from medicalvision.transforms import get_eval_transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedicalVision AI Model")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--backbone", type=str, default="densenet121")
    parser.add_argument("--max_test_samples", type=int, default=100)
    parser.add_argument("--max_bbox_samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("       MEDICALVISION AI - MODEL BENCHMARK & EVALUATION")
    print("=" * 65)

    ckpt_path = Path(args.checkpoint)
    has_ckpt = ckpt_path.exists()
    print(f"Device:           {args.device}")
    print(f"Checkpoint:       {ckpt_path if has_ckpt else 'ImageNet Pretrained Baseline (no custom checkpoint found)'}")

    # Build model
    model = build_model(
        backbone=args.backbone,
        pretrained=True if not has_ckpt else False,
        checkpoint_path=str(ckpt_path) if has_ckpt else None,
        device=args.device,
    )
    model.eval()

    # Load test split
    print("\n[1. Preparing Test Split...]")
    df = load_dataset_metadata(METADATA_FILE, IMAGE_DIR, existing_images_only=True)
    _, _, test_df = split_by_patient(df, train_ratio=0.75, val_ratio=0.10, test_ratio=0.15)

    if args.max_test_samples and args.max_test_samples < len(test_df):
        test_df = test_df.sample(n=args.max_test_samples, random_state=42).reset_index(drop=True)
    print(f"Evaluating on {len(test_df):,} test X-rays...")

    test_dataset = ChestXrayDataset(test_df, image_dir=IMAGE_DIR, transform=get_eval_transforms())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(args.device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.numpy())

    y_pred_probs = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # 1. Classification Metrics
    metrics = evaluate_dataset_predictions(y_true, y_pred_probs, threshold=DEFAULT_THRESHOLD)

    print("\n" + "=" * 78)
    print(f"{'Condition':<20} | {'Accuracy':<8} | {'Precision':<9} | {'Sensitivity':<11} | {'F1':<6} | {'AUROC'}")
    print("=" * 78)
    for disease in DISEASE_CLASSES:
        m = metrics[disease]
        print(f"{disease:<20} | {m['Accuracy']:<8.2%} | {m['Precision']:<9.2%} | {m['Sensitivity']:<11.2%} | {m['F1']:<6.4f} | {m['AUROC']:.4f}")
    print("=" * 78)
    print(f"{'HAMMING ACCURACY':<20} | {metrics['Hamming_Accuracy']:<8.2%} | (Average correctness per label)")
    print(f"{'MACRO AVERAGE':<20} | {metrics['Macro_Accuracy']:<8.2%} | {metrics['Macro_Precision']:<9.2%} | {'':<11} | {metrics['Macro_F1']:<6.4f} | {metrics['Macro_AUROC']:.4f}")
    print(f"{'MICRO PRECISION':<20} | {'':<8} | {metrics['Micro_Precision']:<9.2%}")
    print(f"{'NORMAL/ABNORMAL':<20} | {metrics['Binary_Abnormal_Accuracy']:<8.2%} | {metrics['Binary_Abnormal_Precision']:<9.2%} | (Binary Detection)")
    print("=" * 78)

    # 2. Localization IoU
    print("\n[2. Evaluating Weakly Supervised Bounding Box Localization...]")
    bbox_df = load_bounding_boxes(BBOX_FILE)
    if not bbox_df.empty:
        loc_metrics = evaluate_localization_iou(
            model=model,
            bbox_df=bbox_df,
            image_dir=IMAGE_DIR,
            device=args.device,
            max_samples=args.max_bbox_samples,
        )
        print(f"Evaluated Bounding Box Ground Truths: {loc_metrics['Evaluated_Samples']}")
        print(f"Mean IoU (Intersection over Union):   {loc_metrics['Mean_IoU']:.4f}")
        print(f"Localization Accuracy @ IoU >= 0.10: {loc_metrics['Detection_Rate_0.10']:.1%}")
        print(f"Localization Accuracy @ IoU >= 0.25: {loc_metrics['Detection_Rate_0.25']:.1%}")
    else:
        print("BBox_List_2017.csv not found; skipped bounding box benchmark.")

    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    import numpy as np
    main()
