"""
Data transformations and augmentations for Chest X-ray processing.
"""

from typing import Tuple
from torchvision import transforms
from PIL import Image

from medicalvision.config import IMAGE_SIZE, NORM_MEAN, NORM_STD


def get_train_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """
    Data augmentations appropriate for medical chest X-rays.
    Avoids vertical flips (anatomically invalid for CXR) while applying
    slight rotations, affine shifts, and color jitter.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_eval_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """
    Deterministic evaluation/inference transform.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def preprocess_image_for_inference(
    image: Image.Image,
    image_size: Tuple[int, int] = IMAGE_SIZE
):
    """
    Preprocess a PIL image for model input:
    - Ensures RGB mode (3 channels)
    - Applies eval transforms
    - Adds batch dimension: (1, 3, H, W)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    eval_tf = get_eval_transforms(image_size)
    tensor = eval_tf(image).unsqueeze(0)
    return tensor
