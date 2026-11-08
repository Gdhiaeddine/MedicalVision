import pandas as pd
from medicalvision.config import DISEASE_CLASSES, METADATA_FILE
from medicalvision.dataset import load_dataset_metadata, split_by_patient
from medicalvision.models.loss import compute_class_weights

print("Loading metadata...")
df = load_dataset_metadata(METADATA_FILE, existing_images_only=False)
print("Splitting patients...")
train_df, val_df, test_df = split_by_patient(df, train_ratio=0.75, val_ratio=0.10, test_ratio=0.15)

print(f"Total Scans: {len(df):,}")
print(f"Train Scans: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

print("\n" + "=" * 70)
print(f"{'Condition':<20} | {'Positives':<10} | {'Prevalence (%)':<15} | {'pos_weight':<10}")
print("=" * 70)

weights = compute_class_weights(train_df)
for idx, d in enumerate(DISEASE_CLASSES):
    pos = int(train_df[d].sum())
    prev = (pos / len(train_df)) * 100
    w = float(weights[idx])
    print(f"{d:<20} | {pos:<10,} | {prev:>6.2f}%         | {w:>6.2f}")
print("=" * 70)
