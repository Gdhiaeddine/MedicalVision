"""
MedicalVision AI - Architecture Benchmark Suite:
Comprehensive Head-to-Head Comparison of Backbones:
1. DenseNet-121 (CheXNet Standard)
2. ResNet-50 (Deep Residual Network)
3. EfficientNet-B0 (Compound Scaled Convolutional Network)

Evaluates:
- Number of Parameters (M)
- Inference Latency per Image on CPU (ms)
- Macro ROC-AUC & Macro PR-AUC
- Macro Sensitivity, Specificity, F1-Score
- Screening Specificity and False Positives
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression

from medicalvision.config import DISEASE_CLASSES, CHECKPOINT_DIR, OUTPUT_DIR, METADATA_FILE
from medicalvision.dataset import load_dataset_metadata, split_by_patient, ChestXrayDataset, parse_labels
from medicalvision.transforms import get_train_transforms, get_eval_transforms
from medicalvision.models.classifier import build_model
from medicalvision.calibration import PlattScaler, compute_ece
from medicalvision.thresholding import find_optimal_thresholds, compute_metrics_at_threshold


def train_backbone(
    backbone_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weights: torch.Tensor,
    num_classes: int = len(DISEASE_CLASSES),
    epochs: int = 2,
    lr: float = 0.001,
    save_path: Path = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Fine-tune classification head and final block for given backbone."""
    print(f"\n--- Training Backbone: {backbone_name.upper()} ---")
    model = build_model(backbone=backbone_name, num_classes=num_classes, pretrained=True, device="cpu")
    model.to(device)

    # Freeze earlier layers
    if backbone_name == "densenet121":
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.features.denseblock4.parameters():
            p.requires_grad = True
    elif backbone_name == "resnet50":
        for p in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]:
            for param in p.parameters():
                param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif backbone_name == "efficientnet_b0":
        for p in model.features.parameters():
            p.requires_grad = False
        # Unfreeze last two blocks of features (blocks 7 & 8)
        for p in model.features[-2:].parameters():
            p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params / 1e6:.2f}M | Trainable Params: {trainable_params / 1e6:.2f}M")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, targets).item() * len(imgs)
        val_loss /= len(val_loader.dataset)

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "val_loss": best_val_loss}, save_path)
                print(f"  --> Checkpoint saved to {save_path}")

    return model


def benchmark_all_backbones(
    n_train: int = 300,
    n_val: int = 150,
    n_test: int = 400,
):
    print("=" * 80)
    print(" MedicalVision AI — Backbone Architecture Head-to-Head Benchmark")
    print(" Architectures: DenseNet-121 vs ResNet-50 vs EfficientNet-B0")
    print("=" * 80)

    device = torch.device("cpu")

    # 1. Zero-Leakage Split
    full_df = load_dataset_metadata(METADATA_FILE)
    tr_df, val_df, te_df = split_by_patient(full_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    tr_sub = tr_df.sample(n=min(n_train, len(tr_df)), random_state=42).reset_index(drop=True)
    val_sub = val_df.sample(n=min(n_val, len(val_df)), random_state=42).reset_index(drop=True)
    test_sub = te_df.sample(n=min(n_test, len(te_df)), random_state=42).reset_index(drop=True)

    tr_loader = DataLoader(ChestXrayDataset(tr_sub, transform=get_train_transforms()), batch_size=16, shuffle=True)
    val_loader = DataLoader(ChestXrayDataset(val_sub, transform=get_eval_transforms()), batch_size=16, shuffle=False)
    test_loader = DataLoader(ChestXrayDataset(test_sub, transform=get_eval_transforms()), batch_size=16, shuffle=False)

    print(f"Cohorts: Train={len(tr_sub)} | Val={len(val_sub)} | Test={len(test_sub)}")

    # Positive weights
    y_tr = np.array([parse_labels(lbl) for lbl in tr_sub["Finding Labels"]])
    pos_c = np.sum(y_tr, axis=0)
    pos_w = np.clip((len(tr_sub) - pos_c) / np.maximum(pos_c, 1.0), 1.0, 15.0)
    pos_w_tensor = torch.tensor(pos_w, dtype=torch.float32)

    models_config = {
        "densenet121": {
            "name": "DenseNet-121",
            "ckpt": CHECKPOINT_DIR / "best_model.pth",
        },
        "resnet50": {
            "name": "ResNet-50",
            "ckpt": CHECKPOINT_DIR / "best_model_resnet50.pth",
        },
        "efficientnet_b0": {
            "name": "EfficientNet-B0",
            "ckpt": CHECKPOINT_DIR / "best_model_efficientnet_b0.pth",
        },
    }

    trained_models = {}
    for b_key, cfg in models_config.items():
        ckpt_p = cfg["ckpt"]
        if not ckpt_p.exists():
            trained_m = train_backbone(b_key, tr_loader, val_loader, pos_w_tensor, save_path=ckpt_p, epochs=2)
            trained_models[b_key] = trained_m
        else:
            print(f"\nLoading existing trained weights for {cfg['name']} from {ckpt_p}")
            m = build_model(backbone=b_key, num_classes=len(DISEASE_CLASSES), pretrained=False)
            ckpt_obj = torch.load(ckpt_p, map_location=device, weights_only=False)
            state = ckpt_obj["model_state_dict"] if "model_state_dict" in ckpt_obj else ckpt_obj
            m.load_state_dict(state)
            m.to(device).eval()
            trained_models[b_key] = m

    # 2. Benchmark on Test Set
    print("\n" + "=" * 80)
    print(" Evaluating Models on Locked Test Cohort (400 scans)...")
    print("=" * 80)

    benchmark_results = {}

    def extract_preds(m):
        m.eval()
        logits_l, targets_l = [], []
        # Measure latency
        t0 = time.time()
        with torch.no_grad():
            for imgs, targets, _ in test_loader:
                l = m(imgs.to(device))
                logits_l.append(l.cpu())
                targets_l.append(targets.cpu())
        latency_ms = (time.time() - t0) * 1000 / len(test_sub)
        return torch.cat(logits_l, dim=0).numpy(), torch.cat(targets_l, dim=0).numpy(), latency_ms

    for b_key, m in trained_models.items():
        name = models_config[b_key]["name"]
        print(f"Benchmarking {name}...")

        # Count parameters
        param_count_m = sum(p.numel() for p in m.parameters()) / 1e6
        ckpt_size_mb = os.path.getsize(models_config[b_key]["ckpt"]) / (1024 * 1024)

        test_l, test_y, latency_ms = extract_preds(m)

        # Validation extraction for threshold tuning
        val_logits_l, val_y_l = [], []
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                val_logits_l.append(m(imgs.to(device)).cpu())
                val_y_l.append(targets.cpu())
        val_l = torch.cat(val_logits_l, dim=0).numpy()
        val_y = torch.cat(val_y_l, dim=0).numpy()

        # Calibration
        platt = PlattScaler(num_classes=len(DISEASE_CLASSES))
        platt.fit(val_l, val_y)
        val_probs = platt.transform(val_l)
        test_probs = platt.transform(test_l)

        # Metrics per disease
        aurocs, praucs, recalls, specificities, f1s = [], [], [], [], []
        for c in range(len(DISEASE_CLASSES)):
            yt_val = val_y[:, c]
            yp_val = val_probs[:, c]
            yt_test = test_y[:, c]
            yp_test = test_probs[:, c]

            opt_t = find_optimal_thresholds(yt_val, yp_val)
            th = opt_t["constrained_spec75"]
            met = compute_metrics_at_threshold(yt_test, yp_test, th)

            if len(np.unique(yt_test)) > 1:
                auc = float(roc_auc_score(yt_test, yp_test))
                prauc = float(average_precision_score(yt_test, yp_test))
                aurocs.append(auc)
                praucs.append(prauc)
            recalls.append(met["recall"])
            specificities.append(met["specificity"])
            f1s.append(met["f1"])

        # Meta-screening
        val_bin = (val_y.sum(axis=1) > 0).astype(int)
        test_bin = (test_y.sum(axis=1) > 0).astype(int)
        lr = LogisticRegression(C=1.0, solver="lbfgs").fit(val_l, val_bin)
        test_screen_probs = lr.predict_proba(test_l)[:, 1]

        fpr_v, tpr_v, ths_v = roc_curve(val_bin, lr.predict_proba(val_l)[:, 1])
        valid_idxs = np.where(1.0 - fpr_v >= 0.80)[0]
        th_screen = float(ths_v[valid_idxs[np.argmax(tpr_v[valid_idxs])]]) if len(valid_idxs) > 0 else 0.5
        preds_screen = (test_screen_probs >= th_screen).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_bin, preds_screen).ravel()

        benchmark_results[b_key] = {
            "name": name,
            "parameter_count_million": round(param_count_m, 2),
            "checkpoint_size_mb": round(ckpt_size_mb, 1),
            "cpu_inference_latency_ms": round(latency_ms, 1),
            "macro_roc_auc": round(float(np.mean(aurocs)), 4),
            "macro_pr_auc": round(float(np.mean(praucs)), 4),
            "macro_recall": round(float(np.mean(recalls)), 4),
            "macro_specificity": round(float(np.mean(specificities)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "screening_specificity": round(float(tn / (tn + fp)), 4),
            "screening_sensitivity": round(float(tp / (tp + fn)), 4),
            "screening_false_positives": int(fp),
            "screening_auc": round(float(roc_auc_score(test_bin, test_screen_probs)), 4),
        }

    out_file = OUTPUT_DIR / "backbone_architecture_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 80)
    print(" BACKBONE ARCHITECTURE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<26} | {'DenseNet-121':<15} | {'ResNet-50':<15} | {'EfficientNet-B0':<15}")
    print("-" * 80)
    b_d = benchmark_results["densenet121"]
    b_r = benchmark_results["resnet50"]
    b_e = benchmark_results["efficientnet_b0"]

    print(f"{'Parameters (Millions)':<26} | {b_d['parameter_count_million']:>15.2f} | {b_r['parameter_count_million']:>15.2f} | {b_e['parameter_count_million']:>15.2f}")
    print(f"{'Checkpoint Size (MB)':<26} | {b_d['checkpoint_size_mb']:>15.1f} | {b_r['checkpoint_size_mb']:>15.1f} | {b_e['checkpoint_size_mb']:>15.1f}")
    print(f"{'CPU Latency / scan (ms)':<26} | {b_d['cpu_inference_latency_ms']:>15.1f} | {b_r['cpu_inference_latency_ms']:>15.1f} | {b_e['cpu_inference_latency_ms']:>15.1f}")
    print(f"{'Macro ROC-AUC':<26} | {b_d['macro_roc_auc']:>15.4f} | {b_r['macro_roc_auc']:>15.4f} | {b_e['macro_roc_auc']:>15.4f}")
    print(f"{'Macro PR-AUC':<26} | {b_d['macro_pr_auc']:>15.4f} | {b_r['macro_pr_auc']:>15.4f} | {b_e['macro_pr_auc']:>15.4f}")
    print(f"{'Macro Recall':<26} | {b_d['macro_recall']*100:>14.2f}% | {b_r['macro_recall']*100:>14.2f}% | {b_e['macro_recall']*100:>14.2f}%")
    print(f"{'Macro Specificity':<26} | {b_d['macro_specificity']*100:>14.2f}% | {b_r['macro_specificity']*100:>14.2f}% | {b_e['macro_specificity']*100:>14.2f}%")
    print(f"{'Macro F1-Score':<26} | {b_d['macro_f1']:>15.4f} | {b_r['macro_f1']:>15.4f} | {b_e['macro_f1']:>15.4f}")
    print("-" * 80)
    print(f"{'Screening False Positives':<26} | {b_d['screening_false_positives']:>15} | {b_r['screening_false_positives']:>15} | {b_e['screening_false_positives']:>15}")
    print(f"{'Screening Specificity':<26} | {b_d['screening_specificity']*100:>14.2f}% | {b_r['screening_specificity']*100:>14.2f}% | {b_e['screening_specificity']*100:>14.2f}%")
    print(f"{'Screening Sensitivity':<26} | {b_d['screening_sensitivity']*100:>14.2f}% | {b_r['screening_sensitivity']*100:>14.2f}% | {b_e['screening_sensitivity']*100:>14.2f}%")
    print("=" * 80)
    print(f"Benchmark results saved to {out_file}")

    return benchmark_results


if __name__ == "__main__":
    benchmark_all_backbones()
