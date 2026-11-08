"""
MedicalVision AI - Configuration for the Recommended 5-Class Specialized Model.
Pathologies with superior clinical discriminability:
1. Pneumothorax
2. Edema
3. Consolidation
4. Effusion
5. Cardiomegaly
"""

from pathlib import Path
from medicalvision.config import WORKSPACE_DIR, DATA_DIR, IMAGE_DIR, METADATA_FILE, OUTPUT_DIR, CHECKPOINT_DIR

# The 5 Top-Performing Clinical Conditions
FIVE_DISEASE_CLASSES = [
    "Pneumothorax",
    "Edema",
    "Consolidation",
    "Effusion",
    "Cardiomegaly",
]

NUM_5_CLASSES = len(FIVE_DISEASE_CLASSES)

CHECKPOINT_5CLASS = CHECKPOINT_DIR / "best_model_5class.pth"
COMPARISON_REPORT = OUTPUT_DIR / "5class_vs_14class_comparison.json"
