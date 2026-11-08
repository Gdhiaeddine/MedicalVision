"""
Unit tests for MedicalVision AI dataset parsing and patient leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from medicalvision.config import DISEASE_CLASSES
from medicalvision.dataset import parse_labels, split_by_patient, ChestXrayDataset


def test_parse_labels_no_finding():
    label_str = "No Finding"
    vector = parse_labels(label_str)
    assert len(vector) == len(DISEASE_CLASSES)
    assert np.all(vector == 0.0)


def test_parse_labels_multiple_pathologies():
    label_str = "Cardiomegaly|Effusion|Pneumothorax"
    vector = parse_labels(label_str)
    assert len(vector) == len(DISEASE_CLASSES)

    cardiomegaly_idx = DISEASE_CLASSES.index("Cardiomegaly")
    effusion_idx = DISEASE_CLASSES.index("Effusion")
    pneumothorax_idx = DISEASE_CLASSES.index("Pneumothorax")

    assert vector[cardiomegaly_idx] == 1.0
    assert vector[effusion_idx] == 1.0
    assert vector[pneumothorax_idx] == 1.0

    # Atelectasis should be 0
    atelectasis_idx = DISEASE_CLASSES.index("Atelectasis")
    assert vector[atelectasis_idx] == 0.0


def test_patient_level_split_no_leakage():
    # Synthetic patient metadata
    data = []
    for p_id in range(1, 101):
        # Each patient has 1 to 5 images
        n_images = np.random.randint(1, 5)
        for img_num in range(n_images):
            data.append({
                "Image Index": f"{p_id:08d}_{img_num:03d}.png",
                "Patient ID": p_id,
                "Finding Labels": "Cardiomegaly" if p_id % 3 == 0 else "No Finding",
            })
    df = pd.DataFrame(data)

    train_df, val_df, test_df = split_by_patient(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42)

    # Check that patient IDs do NOT intersect
    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])

    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0
