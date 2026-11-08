"""
MedicalVision AI - Object Detection Dataset for NIH ChestX-ray14 Bounding Boxes.
Parses CXR8/BBox_List_2017.csv (984 boxes across 8 pathologies).
Ensures zero-leakage patient-level train/test split.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from medicalvision.config import (
    IMAGE_DIR,
    BBOX_FILE,
    DETECTION_CLASSES,
    DETECTION_IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
)

CLASS_TO_IDX = {name: idx + 1 for idx, name in enumerate(DETECTION_CLASSES)}
IDX_TO_CLASS = {idx + 1: name for idx, name in enumerate(DETECTION_CLASSES)}


class NIHBBoxDataset(Dataset):
    """
    Supervised object detection dataset for Chest X-ray abnormalities.
    Coordinates converted from NIH [x, y, w, h] to PyTorch [x1, y1, x2, y2].
    """

    def __init__(
        self,
        image_indices: List[str],
        bbox_df: pd.DataFrame,
        image_dir: Path = IMAGE_DIR,
        target_size: Tuple[int, int] = DETECTION_IMAGE_SIZE,
        is_training: bool = False,
    ):
        self.image_indices = sorted(list(set(image_indices)))
        self.bbox_df = bbox_df
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.is_training = is_training

        # Pre-group annotations by Image Index
        self.grouped = self.bbox_df.groupby("Image Index")

    def __len__(self) -> int:
        return len(self.image_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_name = self.image_indices[idx]
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Retrieve all boxes for this image
        rows = self.grouped.get_group(img_name)
        boxes = []
        labels = []

        scale_x = self.target_size[0] / orig_w
        scale_y = self.target_size[1] / orig_h

        for _, row in rows.iterrows():
            finding = str(row["Finding Label"]).strip()
            # Handle potential naming differences like Infiltrate vs Infiltration
            if finding == "Infiltration":
                finding = "Infiltrate"
            if finding not in CLASS_TO_IDX:
                continue

            x = float(row["Bbox [x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h]"])

            # Convert to [x1, y1, x2, y2] and rescale
            x1 = max(0.0, min(float(self.target_size[0]), x * scale_x))
            y1 = max(0.0, min(float(self.target_size[1]), y * scale_y))
            x2 = max(0.0, min(float(self.target_size[0]), (x + w) * scale_x))
            y2 = max(0.0, min(float(self.target_size[1]), (y + h) * scale_y))

            # Ensure valid box dimensions
            if x2 > x1 + 2 and y2 > y1 + 2:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[finding])

        # Resize image
        image = TF.resize(image, list(self.target_size))
        
        # Training horizontal flip augmentation
        if self.is_training and np.random.rand() > 0.5:
            image = TF.hflip(image)
            flipped_boxes = []
            for b in boxes:
                # [x1, y1, x2, y2] -> [W - x2, y1, W - x1, y2]
                w_img = float(self.target_size[0])
                new_x1 = w_img - b[2]
                new_x2 = w_img - b[0]
                flipped_boxes.append([new_x1, b[1], new_x2, b[3]])
            boxes = flipped_boxes

        image_tensor = TF.to_tensor(image)

        if len(boxes) == 0:
            # Degenerate case guard: dummy background box
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "image_name": img_name,
        }

        return image_tensor, target


def get_detection_split(
    bbox_csv: Path = BBOX_FILE,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[NIHBBoxDataset, NIHBBoxDataset, pd.DataFrame]:
    """
    Creates patient-level split for NIH bounding box dataset.
    """
    df = pd.read_csv(bbox_csv)
    # Extract Patient ID (e.g. '00013118' from '00013118_008.png')
    df["Patient ID"] = df["Image Index"].apply(lambda s: s.split("_")[0])

    unique_patients = df["Patient ID"].unique()
    rng = np.random.RandomState(random_seed)
    rng.shuffle(unique_patients)

    n_train = int(len(unique_patients) * train_ratio)
    train_patients = set(unique_patients[:n_train])
    test_patients = set(unique_patients[n_train:])

    train_df = df[df["Patient ID"].isin(train_patients)]
    test_df = df[df["Patient ID"].isin(test_patients)]

    train_images = list(train_df["Image Index"].unique())
    test_images = list(test_df["Image Index"].unique())

    train_dataset = NIHBBoxDataset(train_images, df, is_training=True)
    test_dataset = NIHBBoxDataset(test_images, df, is_training=False)

    return train_dataset, test_dataset, df


def detection_collate_fn(batch):
    """Custom collate function required by TorchVision detection models."""
    return tuple(zip(*batch))
