"""
Standalone training script for MedicalVision AI Chest X-ray model.
Supports patient-level stratified splitting, class-weighted training, and checkpointing.
"""

import argparse
from pathlib import Path
import torch

from medicalvision.config import (
    DISEASE_CLASSES,
    IMAGE_DIR,
    METADATA_FILE,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from medicalvision.dataset import load_dataset_metadata, split_by_patient, get_data_loaders
from medicalvision.models.classifier import build_model
from medicalvision.models.loss import compute_class_weights
from medicalvision.trainer import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedicalVision AI Model")
    parser.add_argument("--backbone", type=str, default="densenet121", choices=["densenet121", "resnet50"])
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max dataset subset for fast training")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch for benchmarking")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("       MEDICALVISION AI - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device:           {args.device}")
    print(f"Backbone:         {args.backbone}")
    print(f"Target Pathologies: {len(DISEASE_CLASSES)} classes")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")

    # 1. Load dataset metadata
    print("\n[Loading dataset metadata...]")
    df = load_dataset_metadata(METADATA_FILE, IMAGE_DIR, existing_images_only=True)
    print(f"Total available images on disk: {len(df):,}")

    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled dataset to {len(df):,} images for fast iteration.")

    # 2. Patient-level split
    print("[Splitting by Patient ID to prevent data leakage...]")
    train_df, val_df, test_df = split_by_patient(df, train_ratio=0.75, val_ratio=0.10, test_ratio=0.15)
    print(f"Train samples: {len(train_df):,} | Val samples: {len(val_df):,} | Test samples: {len(test_df):,}")

    # Verify 0 patient overlap
    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage detected!"
    assert len(train_patients.intersection(val_patients)) == 0, "Patient leakage detected!"
    print("Zero patient leakage verified across splits.")

    # 3. Compute empirical positive class weights
    print("[Computing class weights for imbalanced BCE loss...]")
    pos_weights = compute_class_weights(train_df)
    for disease, w in zip(DISEASE_CLASSES, pos_weights.tolist()):
        print(f"  - {disease:<20}: pos_weight = {w:.2f}")

    # 4. DataLoaders
    loaders = get_data_loaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_dir=IMAGE_DIR,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # 5. Initialize model
    print(f"\n[Initializing {args.backbone} backbone with ImageNet pretrained features...]")
    model = build_model(backbone=args.backbone, pretrained=True, device=args.device)

    # 6. Train model
    history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=args.epochs,
        lr=args.lr,
        pos_weight=pos_weights,
        device=args.device,
        checkpoint_dir=CHECKPOINT_DIR,
        max_batches_per_epoch=args.max_batches,
    )

    print("\nTraining completed successfully!")
    print(f"Best checkpoint saved to: {CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
