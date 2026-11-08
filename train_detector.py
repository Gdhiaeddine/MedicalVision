"""
MedicalVision AI - Train Faster R-CNN on NIH ChestX-ray14 Bounding Box Dataset.
Trains ResNet-50-FPN on 8 NIH annotated pathologies with patient-level split.
"""

import os
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from medicalvision.config import CHECKPOINT_DIR, BBOX_FILE, DETECTION_CLASSES
from medicalvision.detection_dataset import (
    get_detection_split,
    detection_collate_fn,
)
from medicalvision.models.detector import create_faster_rcnn_detector


def train_detector(
    num_epochs: int = 2,
    batch_size: int = 2,
    learning_rate: float = 0.001,
    max_train_samples: int = 150,
    save_path: Path = CHECKPOINT_DIR / "faster_rcnn_nih.pth",
):
    print("=" * 60)
    print(" MedicalVision AI — Faster R-CNN Object Detection Training")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"Device: {device} (Intel CPU optimized)")

    # Data split (patient-level)
    train_dataset, test_dataset, bbox_df = get_detection_split(train_ratio=0.8, random_seed=42)
    
    # Subsample training images if specified
    if max_train_samples and max_train_samples < len(train_dataset):
        indices = list(range(max_train_samples))
        train_sub = torch.utils.data.Subset(train_dataset, indices)
    else:
        train_sub = train_dataset

    print(f"Train images: {len(train_sub)} | Test images: {len(test_dataset)}")
    print(f"Pathology classes ({len(DETECTION_CLASSES)}): {DETECTION_CLASSES}")

    train_loader = DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=0,
    )

    # Initialize model with frozen ResNet-50 backbone for fast & stable fine-tuning on CPU
    print("Initializing Faster R-CNN with ResNet-50-FPN backbone...")
    model = create_faster_rcnn_detector(
        num_classes=len(DETECTION_CLASSES) + 1,
        pretrained_backbone=True,
        trainable_backbone_layers=0,
    )
    model.to(device)

    # Parameters to optimize: exclude frozen early layers
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    best_loss = float("inf")
    os.makedirs(save_path.parent, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        batch_count = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # Faster R-CNN computes multi-task losses automatically in train mode
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            epoch_loss += losses.item()
            batch_count += 1
            if batch_count % 30 == 0:
                print(f"  Epoch [{epoch}/{num_epochs}] Step [{batch_count}/{len(train_loader)}] Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / max(1, batch_count)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{num_epochs} completed in {elapsed:.1f}s | Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "classes": DETECTION_CLASSES,
                },
                save_path,
            )
            print(f"  --> Checkpoint saved to {save_path} (Loss: {best_loss:.4f})")

    print("\nTraining completed successfully!")
    return save_path


if __name__ == "__main__":
    train_detector(num_epochs=3, batch_size=4, learning_rate=0.001)
