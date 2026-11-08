---
title: MedicalVision AI
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app_gradio.py
---

# 🩻 MedicalVision AI
**Deep Learning Chest Radiograph Abnormality Screening, Decoupled Multi-Pathology Classification & Grad-CAM Anatomical Localization**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> [!IMPORTANT]
> **Research Prototype Notice**: MedicalVision AI is developed strictly for research, benchmarking, and algorithmic explainability demonstration on the NIH ChestX-ray14 dataset. It is **not an FDA/CE-cleared diagnostic device** and should never replace certified clinical judgment.

---

## 🌟 Key Capabilities

- **🎯 Decoupled High-Signal 5-Class Classifier**: Specialized deep neural network fine-tuned with **Asymmetric Anatomical Contrastive Loss** (`DisentangledBCEWithLogitsLoss`) to cleanly separate mutually confusable or comorbid conditions (e.g. *Cardiomegaly* vs. *Pleural Effusion* and *Consolidation*).
- **🔬 Visual Explainability (Grad-CAM)**: Regional pixel gradient attribution with automated contour detection and salient bounding boxes.
- **⚖️ Cardio-Pleural Differential Diagnosis Module**: Automated separation metric ($\Delta_{\text{cardio-eff}}$) and dual-panel synchronized Grad-CAM (Mediastinal cardiac border vs. Costophrenic sulci).
- **🎯 Supervised Object Detector (Faster R-CNN)**: ResNet-50-FPN object detection trained directly on NIH expert bounding box annotations.
- **🛡️ Uncertainty Quantification (MC Dropout)**: Epistemic uncertainty estimation via Monte Carlo forward passes to flag borderline/ambiguous cases.
- **⚖️ Rigorous Probability Calibration**: Platt Scaling and temperature calibration ensuring positive Brier Skill Scores ($BSS > 0$) across all target classes.
- **📊 14-Class Global Multi-Label CheXNet Baseline**: Full 14-disease chest pathology spectrum classification.

---

## 🏗️ Architecture & Supported Backbones

| Backbone Architecture | Parameters | Key Architectural Characteristics |
|---|:---:|---|
| **DenseNet-121** (CheXNet standard) | 7.0M | Dense connectivity preserving low-level edge features and fine thoracic textures |
| **ConvNeXt-Tiny** (Modern Pure-CNN) | 28.6M | $7 \times 7$ depthwise convolutions, inverted bottlenecks, high spatial discrimination |
| **ResNet-50** (Residual Networks) | 23.5M | Classic bottleneck residual blocks for deep representation |
| **EfficientNet-B0** (Compound Scaling) | 4.0M | Lightweight compound-scaled architecture for low-latency edge deployment |

---

## 📊 Gold-Standard Clinical Evaluation (500 Locked Test Scans, 405 Unique Patients)

Evaluation performed on a **zero-leakage, patient-disjoint held-out cohort** with **3,000 paired clustered patient bootstrap repetitions**:

| Target Pathology | Test Positives ($N_{\text{pos}}$) | Test Negatives ($N_{\text{neg}}$) | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Effusion** | 88 | 412 | **0.7843** [0.730, 0.836] | **0.4099** [0.322, 0.528] | **64.77 %** [54.6 %, 73.8 %] | **77.91 %** [73.5 %, 82.1 %] |
| **Pneumothorax** | 27 | 473 | **0.8178** [0.743, 0.890] | **0.3294** [0.180, 0.514] | **62.96 %** [44.4 %, 80.0 %] | **77.38 %** [73.3 %, 81.4 %] |
| **Edema** | 13 | 487 | **0.9106** [0.780, 0.974] | **0.2426** [0.103, 0.448] | **92.31 %** [70.5 %, 100.0 %] | **67.56 %** [62.7 %, 72.7 %] |
| **Consolidation** | 26 | 474 | **0.6913** [0.583, 0.780] | **0.1385** [0.062, 0.256] | **57.69 %** [37.5 %, 76.2 %] | **70.46 %** [65.8 %, 74.9 %] |
| **Cardiomegaly** | 23 | 477 | **0.7245** [0.575, 0.847] | **0.1866** [0.074, 0.401] | **56.52 %** [34.6 %, 79.2 %] | **75.26 %** [71.0 %, 79.2 %] |

- **Macro ROC-AUC**: **0.7857** (+8.8 pp gain over baseline)
- **Macro PR-AUC**: **0.2614** (+74.3% relative gain)
- **Macro Sensitivity**: **66.85 %** (+13.9 pp gain)
- **Cardiomegaly / Effusion Spatial Overlap**: Pearson $r = -0.577$ (statistically orthogonal spatial attention)

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Gdhiaeddine/MedicalVision.git
cd MedicalVision
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Interactive Streamlit Web Application

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run Automated Unit Tests

```bash
python -m pytest tests/
```

### 4. Reproduce the Locked Evaluation Suite

```bash
python run_locked_final_evaluation.py
```

---

## 📁 Repository Structure

```
MedicalVision/
├── app.py                                   # Full Streamlit Interactive Web Application
├── train_5class_model.py                    # Stratified Disentangled 5-Class Training Pipeline
├── run_locked_final_evaluation.py           # 3,000 Clustered Patient Bootstrap Evaluation Suite
├── run_5class_comparison.py                 # Head-to-Head 5-Class vs 14-Class Comparative Benchmark
├── requirements.txt                         # Python Dependencies
├── pytest.ini                              # Pytest Configuration
├── .gitignore                               # Large file, cache, and dataset exclusions
├── medicalvision/                           # Core Library
│   ├── config.py                            # Hyperparameters, Paths, and Class Constants
│   ├── config_5class.py                     # 5-Class Target Definitions and Checkpoint Paths
│   ├── dataset.py                           # Master Patient Splitting and Metadata Parsing
│   ├── dataset_5class.py                    # Counter-Factual Disentangled Cohort Sampler
│   ├── loss.py                              # DisentangledBCEWithLogitsLoss (Contrastive Margin)
│   ├── transforms.py                        # Augmentation and Inference Preprocessing
│   ├── calibration.py                       # Platt Scaling and Temperature Scaler Engine
│   ├── uncertainty.py                       # Monte Carlo Dropout Uncertainty Quantification
│   ├── report_generator.py                  # Structured Diagnostic Radiology Text Reporter
│   ├── thresholding.py                      # Multi-Threshold Screening Optimization
│   ├── bootstrap.py                         # Clustered Patient-Level Paired Bootstrap Engine
│   ├── models/
│   │   └── classifier.py                    # Multi-Backbone Classifier (DenseNet, ConvNeXt, ResNet)
│   └── interpretability/
│       └── gradcam.py                       # Gradient-weighted Class Activation Mapping
├── outputs/                                 # Evaluation Reports and Plots
│   ├── locked_gold_standard_evaluation.json # Certified Evaluation Metrics
│   └── locked_screening_curves.png          # ROC and PR Curves Plot
└── tests/                                   # Automated Pytest Suite (21 Unit Tests)
    ├── test_dataset.py
    ├── test_gradcam.py
    ├── test_loss.py
    ├── test_model.py
    ├── test_uncertainty.py
    └── test_v2_features.py
```

---

## 📜 Citation & Acknowledgments
Dataset provided by the NIH Clinical Center:
- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* IEEE CVPR 2017.
