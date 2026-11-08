"""
NIH ChestX-ray14 Dataset Parser and PyTorch Dataset implementation.
Ensures patient-level leakage-free train/val/test splits.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from medicalvision.config import (
    DISEASE_CLASSES,
    IMAGE_DIR,
    METADATA_FILE,
    BBOX_FILE,
    TRAIN_VAL_LIST,
    TEST_LIST,
    BATCH_SIZE,
    NUM_WORKERS,
)
from medicalvision.transforms import get_train_transforms, get_eval_transforms


def parse_labels(label_str: str, disease_classes: List[str] = DISEASE_CLASSES) -> np.ndarray:
    """
    Convert a pipe-separated pathology label string into a multi-hot binary vector of length 14.
    If 'No Finding', all 14 entries will be 0.
    """
    vector = np.zeros(len(disease_classes), dtype=np.float32)
    if not isinstance(label_str, str) or label_str.strip() == "No Finding":
        return vector

    present_findings = {f.strip() for f in label_str.split("|")}
    for idx, disease in enumerate(disease_classes):
        # Match exact string, or underscore/space variants
        d_space = disease.replace("_", " ")
        d_under = disease.replace(" ", "_")
        if disease in present_findings or d_space in present_findings or d_under in present_findings:
            vector[idx] = 1.0
    return vector


def load_dataset_metadata(
    metadata_csv: Path = METADATA_FILE,
    image_dir: Path = IMAGE_DIR,
    existing_images_only: bool = True,
) -> pd.DataFrame:
    """
    Load NIH ChestX-ray metadata and verify image file availability.
    Adds binary indicator columns for each disease and a multi-hot label array.
    """
    df = pd.read_csv(metadata_csv)

    # Filter to images that actually exist on disk if required
    if existing_images_only and os.path.exists(image_dir):
        existing_filenames = set(os.listdir(image_dir))
        df = df[df["Image Index"].isin(existing_filenames)].reset_index(drop=True)

    # Vectorized multi-hot encoding across all 14 conditions (fast C implementation)
    finding_labels = df["Finding Labels"].fillna("No Finding")
    for disease in DISEASE_CLASSES:
        # Match both underscore and space versions
        d_space = disease.replace("_", " ")
        d_under = disease.replace(" ", "_")
        pattern = f"{d_space}|{d_under}"
        df[disease] = finding_labels.str.contains(pattern, case=False, regex=True).astype(np.float32)

    df["Is_Abnormal"] = (df[DISEASE_CLASSES].sum(axis=1) > 0).astype(int)
    return df


def split_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-level stratified train / validation / test split.
    Guarantees NO patient ID appears in more than one split to prevent data leakage.
    """
    rng = np.random.RandomState(random_seed)
    unique_patients = df["Patient ID"].unique()
    rng.shuffle(unique_patients)

    n_patients = len(unique_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train : n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val :])

    train_df = df[df["Patient ID"].isin(train_patients)].copy().reset_index(drop=True)
    val_df = df[df["Patient ID"].isin(val_patients)].copy().reset_index(drop=True)
    test_df = df[df["Patient ID"].isin(test_patients)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def load_bounding_boxes(bbox_csv: Path = BBOX_FILE) -> pd.DataFrame:
    """
    Load ground-truth bounding box annotations from BBox_List_2017.csv.
    Standardizes column names and bounding box format: [x, y, w, h].
    """
    if not os.path.exists(bbox_csv):
        return pd.DataFrame()
    bbox_df = pd.read_csv(bbox_csv)
    # The CSV header is often 'Image Index,Finding Label,Bbox [x,y,w,h],,,'
    # Clean up column names
    cols = ["Image_Index", "Finding_Label", "x", "y", "w", "h"]
    bbox_df = bbox_df.iloc[:, :6]
    bbox_df.columns = cols
    bbox_df["Finding_Label"] = bbox_df["Finding_Label"].str.replace(" ", "_")
    return bbox_df


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray images.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path = IMAGE_DIR,
        transform=None,
        disease_classes: List[str] = DISEASE_CLASSES,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.disease_classes = disease_classes

        # Pre-extract target matrix for instant indexing
        self.targets = self.df[self.disease_classes].values.astype(np.float32)
        self.image_filenames = self.df["Image Index"].values
        self.patient_ids = self.df["Patient ID"].values if "Patient ID" in self.df.columns else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.image_filenames[idx]
        image_path = self.image_dir / filename

        # Load image (NIH images are Grayscale PNGs, convert to 3-channel RGB)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = get_eval_transforms()(image)

        label_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        return image_tensor, label_tensor, filename


def get_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    image_dir: Path = IMAGE_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and optional test sets.
    """
    train_dataset = ChestXrayDataset(
        df=train_df,
        image_dir=image_dir,
        transform=get_train_transforms(),
    )
    val_dataset = ChestXrayDataset(
        df=val_df,
        image_dir=image_dir,
        transform=get_eval_transforms(),
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }

    if test_df is not None:
        test_dataset = ChestXrayDataset(
            df=test_df,
            image_dir=image_dir,
            transform=get_eval_transforms(),
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    return loaders
