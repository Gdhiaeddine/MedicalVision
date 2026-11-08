"""
MedicalVision AI - Dataset Pipeline for the 5-Class High-Confidence Clinical Cohort.
Retains 'No Finding' (healthy controls) and images containing at least one of the 5 target classes.
Excludes scans with ONLY irrelevant diseases to eliminate label noise and false negative penalties.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from medicalvision.config import IMAGE_DIR, METADATA_FILE
from medicalvision.config_5class import FIVE_DISEASE_CLASSES
from medicalvision.transforms import get_train_transforms, get_eval_transforms
from medicalvision.dataset import split_by_patient


def load_5class_metadata(
    metadata_csv: Path = METADATA_FILE,
    target_classes: List[str] = FIVE_DISEASE_CLASSES,
) -> pd.DataFrame:
    """
    Filters NIH metadata to:
    1. Images with at least one target class
    2. Images with 'No Finding' (pure normal controls)
    """
    df = pd.read_csv(metadata_csv)

    has_target = df["Finding Labels"].apply(lambda s: any(t in str(s) for t in target_classes))
    is_normal = df["Finding Labels"].astype(str).str.strip() == "No Finding"

    filtered_df = df[has_target | is_normal].copy().reset_index(drop=True)

    # Precompute 5-class multi-hot vectors
    def get_5hot(label_str):
        vec = np.zeros(len(target_classes), dtype=np.float32)
        if not isinstance(label_str, str) or label_str.strip() == "No Finding":
            return vec
        findings = {f.strip() for f in label_str.split("|")}
        for i, c in enumerate(target_classes):
            if c in findings:
                vec[i] = 1.0
        return vec

    filtered_df["target_vector"] = filtered_df["Finding Labels"].apply(get_5hot)
    return filtered_df


class FiveClassChestXrayDataset(Dataset):
    """PyTorch Dataset for 5-class target cohort."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path = IMAGE_DIR,
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.image_dir / img_name

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(row["target_vector"], dtype=torch.float32)
        return image, target, img_name


def get_5class_patient_splits(
    metadata_csv: Path = METADATA_FILE,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CRITICAL ZERO-LEAKAGE MASTER SPLIT:
    First partitions ALL patients across the entire dataset.
    Then filters cohorts within each partition to guarantee 100% independence
    across all models (14-class, 5-class, meta-screeners).
    """
    from medicalvision.dataset import load_dataset_metadata, split_by_patient

    df_all = load_dataset_metadata(metadata_csv)
    tr_master, val_master, te_master = split_by_patient(
        df_all, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random_seed=random_seed
    )

    def filter_and_format(df):
        has_target = df["Finding Labels"].apply(lambda s: any(t in str(s) for t in FIVE_DISEASE_CLASSES))
        is_normal = df["Finding Labels"].astype(str).str.strip() == "No Finding"
        filtered = df[has_target | is_normal].copy().reset_index(drop=True)

        def get_5hot(label_str):
            vec = np.zeros(len(FIVE_DISEASE_CLASSES), dtype=np.float32)
            if not isinstance(label_str, str) or label_str.strip() == "No Finding":
                return vec
            findings = {f.strip() for f in label_str.split("|")}
            for i, c in enumerate(FIVE_DISEASE_CLASSES):
                if c in findings:
                    vec[i] = 1.0
            return vec

        filtered["target_vector"] = filtered["Finding Labels"].apply(get_5hot)
        return filtered

    return filter_and_format(tr_master), filter_and_format(val_master), filter_and_format(te_master)


def get_balanced_5class_training_cohort(
    train_df: pd.DataFrame,
    samples_per_class: int = 150,
    normal_samples: int = 220,
    random_seed: int = 42,
    disentangle_cardio_eff: bool = True,
) -> pd.DataFrame:
    """
    Builds a stratified, class-balanced training cohort from the training split.
    Guarantees sufficient positive representation for every single target class
    and explicitly mines counter-factual pairs (Cardiomegaly without Effusion,
    and Effusion without Cardiomegaly) to enforce anatomical boundary disentanglement.
    """
    selected_indices = set()
    rng = np.random.RandomState(random_seed)

    cardio_idx = FIVE_DISEASE_CLASSES.index("Cardiomegaly")
    eff_idx = FIVE_DISEASE_CLASSES.index("Effusion")

    if disentangle_cardio_eff:
        # 1. Pure Cardiomegaly WITHOUT Effusion (sharp costophrenic angles)
        pure_c = train_df[train_df["target_vector"].apply(lambda v: v[cardio_idx] == 1.0 and v[eff_idx] == 0.0)]
        n_c = min(samples_per_class, len(pure_c))
        selected_indices.update(rng.choice(pure_c.index, size=n_c, replace=False))

        # 2. Pure Effusion WITHOUT Cardiomegaly (normal cardiac silhouette)
        pure_e = train_df[train_df["target_vector"].apply(lambda v: v[eff_idx] == 1.0 and v[cardio_idx] == 0.0)]
        n_e = min(samples_per_class, len(pure_e))
        selected_indices.update(rng.choice(pure_e.index, size=n_e, replace=False))

        # 3. Comorbid presentation (Both present, e.g. Congestive Heart Failure)
        both_ce = train_df[train_df["target_vector"].apply(lambda v: v[cardio_idx] == 1.0 and v[eff_idx] == 1.0)]
        n_both = min(int(samples_per_class * 0.5), len(both_ce))
        selected_indices.update(rng.choice(both_ce.index, size=n_both, replace=False))

        # 4. Other pathologies (Pneumothorax, Edema, Consolidation)
        for idx, disease in enumerate(FIVE_DISEASE_CLASSES):
            if idx in (cardio_idx, eff_idx):
                continue
            pos_subset = train_df[train_df["target_vector"].apply(lambda v: v[idx] == 1.0)]
            n_to_pick = min(samples_per_class, len(pos_subset))
            selected_indices.update(rng.choice(pos_subset.index, size=n_to_pick, replace=False))
    else:
        for idx, disease in enumerate(FIVE_DISEASE_CLASSES):
            pos_subset = train_df[train_df["target_vector"].apply(lambda v: v[idx] == 1.0)]
            n_to_pick = min(samples_per_class, len(pos_subset))
            selected_indices.update(rng.choice(pos_subset.index, size=n_to_pick, replace=False))

    # Sample pure normal (No Finding) controls
    normal_subset = train_df[train_df["Finding Labels"].astype(str).str.strip() == "No Finding"]
    n_norm_pick = min(normal_samples, len(normal_subset))
    selected_indices.update(rng.choice(normal_subset.index, size=n_norm_pick, replace=False))

    # Assemble balanced dataframe
    balanced_df = train_df.loc[list(selected_indices)].copy()
    balanced_df = balanced_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return balanced_df
