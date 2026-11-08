"""
Training pipeline for MedicalVision AI ChestXrayClassifier.
Tracks training/validation multi-label loss and mean AUROC across 14 conditions.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from medicalvision.config import CHECKPOINT_DIR, DISEASE_CLASSES, LEARNING_RATE, WEIGHT_DECAY
from medicalvision.models.classifier import ChestXrayClassifier
from medicalvision.models.loss import WeightedBCELoss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> float:
    """
    Run one training epoch over DataLoader.
    """
    model.train()
    running_loss = 0.0
    total_batches = 0

    for idx, (images, targets, _) in enumerate(dataloader):
        if max_batches is not None and idx >= max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    return running_loss / max(1, total_batches)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
    max_batches: Optional[int] = None,
    disease_classes=DISEASE_CLASSES,
) -> Tuple[float, Dict[str, float], float]:
    """
    Evaluate validation loss and per-class AUROC.
    """
    model.eval()
    running_loss = 0.0
    total_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for idx, (images, targets, _) in enumerate(dataloader):
            if max_batches is not None and idx >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item()
            total_batches += 1

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / max(1, total_batches)

    if not all_preds:
        return val_loss, {}, 0.5

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Compute AUROC per disease class
    auroc_per_class = {}
    valid_aurocs = []
    for c_idx, disease in enumerate(disease_classes):
        # AUROC requires at least one positive and one negative sample
        y_true = all_targets[:, c_idx]
        y_score = all_preds[:, c_idx]
        if len(np.unique(y_true)) > 1:
            score = roc_auc_score(y_true, y_score)
            auroc_per_class[disease] = float(score)
            valid_aurocs.append(score)
        else:
            auroc_per_class[disease] = 0.5

    mean_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.5
    return val_loss, auroc_per_class, mean_auroc


def train_model(
    model: ChestXrayClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    pos_weight: Optional[torch.Tensor] = None,
    device: str = "cpu",
    checkpoint_dir: Path = CHECKPOINT_DIR,
    max_batches_per_epoch: Optional[int] = None,
) -> Dict[str, list]:
    """
    Complete training loop with checkpoint saving.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pth"

    criterion = WeightedBCELoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    history = {"train_loss": [], "val_loss": [], "val_mean_auroc": []}
    best_auroc = -1.0

    print(f"Starting training on device: {device} for {epochs} epoch(s)...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=max_batches_per_epoch,
        )

        val_loss, aurocs, mean_auroc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches_per_epoch,
        )

        scheduler.step(mean_auroc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mean_auroc"].append(mean_auroc)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUROC: {mean_auroc:.4f}")

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (AUROC: {best_auroc:.4f})")

    return history
