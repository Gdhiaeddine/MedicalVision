"""
MedicalVision AI - Configuration and Hyperparameters.
Defines pathology classes, file paths, image sizes, and model parameters.
"""

from pathlib import Path

# Base Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "CXR8"
IMAGE_DIR = DATA_DIR / "images" / "images"
METADATA_FILE = DATA_DIR / "Data_Entry_2017_v2020.csv"
BBOX_FILE = DATA_DIR / "BBox_List_2017.csv"
TRAIN_VAL_LIST = DATA_DIR / "train_val_list.txt"
TEST_LIST = DATA_DIR / "test_list.txt"
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# 14 Pathologies defined in NIH ChestX-ray14 & project specification
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES = len(DISEASE_CLASSES)

# 8 Bounding Box Pathologies in BBox_List_2017.csv
DETECTION_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltrate",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]
DETECTION_NUM_CLASSES = len(DETECTION_CLASSES) + 1  # +1 for background class 0
DETECTION_IMAGE_SIZE = (512, 512)

# Image and training hyper-parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_WORKERS = 0  # 0 is safe for Windows multi-processing
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.2
MC_DROPOUT_SAMPLES = 10

# Normalization constants (ImageNet defaults)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Decision thresholds (clinical screening operating points)
DEFAULT_THRESHOLD = 0.5
HIGH_UNCERTAINTY_THRESHOLD = 0.15
