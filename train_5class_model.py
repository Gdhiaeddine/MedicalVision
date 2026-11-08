"""
MedicalVision AI - Train Specialized 5-Class High-Precision Classifier.
Uses Stratified Multi-Label Class-Balanced Sampling (preventing rare-class starvation),
higher spatial resolution (384x384), and modern backbone fine-tuning (ConvNeXt-Tiny or DenseNet-121).
"""

from pathlib import Path
import time
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from medicalvision.config import CHECKPOINT_DIR
from medicalvision.config_5class import FIVE_DISEASE_CLASSES, CHECKPOINT_5CLASS
from medicalvision.dataset_5class import (
    get_5class_patient_splits,
    get_balanced_5class_training_cohort,
    FiveClassChestXrayDataset,
)
from medicalvision.transforms import get_train_transforms, get_eval_transforms
from medicalvision.models.classifier import build_model
from medicalvision.loss import DisentangledBCEWithLogitsLoss


def train_5class_model(
    backbone: str = "densenet121",
    num_epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 4e-4,
    samples_per_class: int = 150,
    normal_samples: int = 220,
    max_val_samples: int = 120,
    image_size: Tuple[int, int] = (224, 224),
    checkpoint_output: Path = CHECKPOINT_5CLASS,
) -> Path:
    """
    Train a specialized 5-class clinical model with stratified balanced sampling.
    """
    print("=" * 70)
    print(f" MedicalVision AI — Training High-Precision 5-Class Model [{backbone.upper()}]")
    print(" Target conditions: " + ", ".join(FIVE_DISEASE_CLASSES))
    print(f" Resolution: {image_size} | Epochs: {num_epochs} | LR: {learning_rate}")
    print("=" * 70)

    device = torch.device("cpu")

    # 1. Dataset splits (Zero-leakage master partition)
    raw_train_df, raw_val_df, _ = get_5class_patient_splits()

    # Stratified Balanced Training Cohort
    train_df = get_balanced_5class_training_cohort(
        raw_train_df,
        samples_per_class=samples_per_class,
        normal_samples=normal_samples,
        random_seed=42,
    )

    if max_val_samples:
        val_df = raw_val_df.sample(n=min(max_val_samples, len(raw_val_df)), random_state=42).reset_index(drop=True)
    else:
        val_df = raw_val_df

    print(f"Balanced Cohort Sizes: Train={len(train_df)} | Val={len(val_df)}")

    # 2. Compute class positive weights (mildly regularized to prevent prior inflation)
    y_train = np.stack(train_df["target_vector"].to_numpy())
    pos_counts = np.sum(y_train, axis=0)
    neg_counts = len(train_df) - pos_counts
    raw_weights = neg_counts / np.maximum(pos_counts, 1.0)
    pos_weights = np.clip(raw_weights * 0.5, 1.0, 2.2)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    print("\nTraining Cohort Class Distribution & Regularized Positive Weights:")
    for i, c in enumerate(FIVE_DISEASE_CLASSES):
        print(f"  - {c:<15}: {int(pos_counts[i]):>3} pos ({pos_counts[i]/len(train_df):.1%}) | PosWeight = {pos_weights[i]:.2f}")

    # Datasets and Loaders
    train_ds = FiveClassChestXrayDataset(train_df, transform=get_train_transforms(image_size=image_size))
    val_ds = FiveClassChestXrayDataset(val_df, transform=get_eval_transforms(image_size=image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Model construction & parameter freezing
    model = build_model(
        backbone=backbone,
        num_classes=len(FIVE_DISEASE_CLASSES),
        pretrained=True,
        device="cpu",
    )

    if backbone in ["convnext_tiny", "convnext"]:
        # ConvNeXt features has 8 stages (0..7). Freeze early stages (0..5), train stages 6, 7 & classifier
        for idx, stage in enumerate(model.features):
            requires_grad = (idx >= 6)
            for param in stage.parameters():
                param.requires_grad = requires_grad
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif backbone == "densenet121":
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.features.denseblock3.parameters():
            param.requires_grad = True
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {backbone} | Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")

    cardio_idx = FIVE_DISEASE_CLASSES.index("Cardiomegaly")
    eff_idx = FIVE_DISEASE_CLASSES.index("Effusion")
    criterion = DisentangledBCEWithLogitsLoss(
        pos_weight=pos_weight_tensor,
        cardio_idx=cardio_idx,
        eff_idx=eff_idx,
        margin=0.08,
        lambda_contrast=2.5,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    checkpoint_output = Path(checkpoint_output)
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)

    print("\nStarting Training Loop...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * len(imgs)
        val_loss /= len(val_ds)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch}/{num_epochs}] ({elapsed:.1f}s) — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": FIVE_DISEASE_CLASSES,
                "backbone": backbone,
                "image_size": image_size,
                "val_loss": best_val_loss,
            }, checkpoint_output)
            print(f"  --> Saved optimal checkpoint to {checkpoint_output} (Val Loss: {best_val_loss:.4f})")

    print("\nTraining completed successfully.")
    return checkpoint_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 5-Class CXR Model")
    parser.add_argument("--backbone", type=str, default="densenet121", choices=["convnext_tiny", "densenet121"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--samples_per_class", type=int, default=150)
    args = parser.parse_args()

    train_5class_model(
        backbone=args.backbone,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
    )
