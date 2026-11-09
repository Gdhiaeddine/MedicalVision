"""
MedicalVision AI - Interactive Clinical Decision Support Dashboard (Streamlit).
Multi-label Chest X-ray Abnormality Detection, Grad-CAM Explainability,
Weakly Supervised Bounding Box Localization, and MC Dropout Uncertainty.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch

# Base Application Paths (robust across local and cloud environments)
WORKSPACE_DIR = Path(__file__).resolve().parent

from medicalvision.config import (
    DISEASE_CLASSES,
    IMAGE_DIR,
    METADATA_FILE,
    BBOX_FILE,
    DEFAULT_THRESHOLD,
    CHECKPOINT_DIR,
)
from medicalvision.config_5class import FIVE_DISEASE_CLASSES
from medicalvision.models.classifier import build_model
from medicalvision.models.detector import CXRDetector
from medicalvision.binary_screener import BinaryScreener
from medicalvision.transforms import preprocess_image_for_inference
from medicalvision.interpretability.gradcam import GradCAM, overlay_heatmap_on_image
from medicalvision.interpretability.localization import extract_bounding_boxes, draw_bounding_boxes_on_image
from medicalvision.uncertainty import estimate_uncertainty_mc_dropout
from medicalvision.report_generator import generate_radiology_report, ANATOMICAL_ZONES

# Page Configuration
st.set_page_config(
    page_title="MedicalVision AI — Chest X-ray Diagnostic Suite",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Clinical Aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400E;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Deep Learning Classifier...")
def get_cached_model(backbone: str = "densenet121"):
    if backbone == "resnet50":
        checkpoint_file = CHECKPOINT_DIR / "best_model_resnet50.pth"
    elif backbone in ["efficientnet_b0", "efficientnet-b0"]:
        checkpoint_file = CHECKPOINT_DIR / "best_model_efficientnet_b0.pth"
    elif backbone == "convnext_tiny":
        checkpoint_file = CHECKPOINT_DIR / "best_model_convnext_tiny.pth"
    else:
        checkpoint_file = CHECKPOINT_DIR / "best_model.pth"

    has_ckpt = checkpoint_file.exists()
    model = build_model(
        backbone=backbone,
        pretrained=True if not has_ckpt else False,
        checkpoint_path=str(checkpoint_file) if has_ckpt else None,
        device="cpu",
    )
    model.eval()
    return model


def ensure_checkpoint_exists(checkpoint_path: Path):
    """Ensure model weights are present (downloads from GitHub if LFS pointer)."""
    need_download = False
    if not checkpoint_path.exists():
        need_download = True
    elif checkpoint_path.stat().st_size < 10000:
        need_download = True

    if need_download:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/Gdhiaeddine/MedicalVision/raw/main/outputs/checkpoints/best_model_5class.pth"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(checkpoint_path))
        except Exception:
            pass


@st.cache_resource(show_spinner="Loading 5-Class High-Signal Classifier...")
def get_cached_5class_model():
    checkpoint_file = CHECKPOINT_DIR / "best_model_5class.pth"
    ensure_checkpoint_exists(checkpoint_file)
    has_ckpt = checkpoint_file.exists() and checkpoint_file.stat().st_size > 10000
    model = build_model(
        backbone="densenet121",
        num_classes=len(FIVE_DISEASE_CLASSES),
        pretrained=False if has_ckpt else True,
        checkpoint_path=str(checkpoint_file) if has_ckpt else None,
        device="cpu",
    )
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading Faster R-CNN Object Detector...")
def get_cached_detector():
    ckpt = CHECKPOINT_DIR / "faster_rcnn_nih.pth"
    detector = CXRDetector(
        checkpoint_path=ckpt if ckpt.exists() else None,
        confidence_threshold=0.03,
        device=torch.device("cpu"),
    )
    return detector



def ensure_sample_images_exist(sample_dir: Path):
    """Ensure sample PNGs are actual image binaries, downloading from GitHub if unpulled LFS pointers."""
    if not sample_dir.exists():
        sample_dir.mkdir(parents=True, exist_ok=True)

    sample_files = [
        "00000001_000.png",
        "00000001_002.png",
        "00000002_000.png",
        "00000008_000.png",
        "00000013_023.png",
        "00000025_000.png",
        "00000032_012.png",
    ]
    for fname in sample_files:
        fpath = sample_dir / fname
        if not fpath.exists() or fpath.stat().st_size < 1000:
            url = f"https://github.com/Gdhiaeddine/MedicalVision/raw/main/assets/sample_scans/{fname}"
            try:
                import urllib.request
                urllib.request.urlretrieve(url, str(fpath))
            except Exception:
                pass


@st.cache_data(show_spinner="Indexing sample X-rays...")
def get_curated_samples():
    """Find available sample images with interesting pathologies from curated assets and local dataset."""
    sample_assets_dir = WORKSPACE_DIR / "assets" / "sample_scans"
    ensure_sample_images_exist(sample_assets_dir)

    # Always include the gold-standard curated reference cases (available in both cloud & local)
    samples = {
        "No Finding (Scan #00000002_000.png)": {
            "filename": "00000002_000.png",
            "path": str(sample_assets_dir / "00000002_000.png") if (sample_assets_dir / "00000002_000.png").exists() else str(IMAGE_DIR / "00000002_000.png"),
            "labels": "No Finding",
            "patient_id": 2,
            "age": 81,
            "sex": "M",
        },
        "Cardiomegaly (Scan #00000001_000.png)": {
            "filename": "00000001_000.png",
            "path": str(sample_assets_dir / "00000001_000.png") if (sample_assets_dir / "00000001_000.png").exists() else str(IMAGE_DIR / "00000001_000.png"),
            "labels": "Cardiomegaly",
            "patient_id": 1,
            "age": 58,
            "sex": "M",
        },
        "Cardiomegaly Pure Test (Scan #00000008_000.png)": {
            "filename": "00000008_000.png",
            "path": str(sample_assets_dir / "00000008_000.png") if (sample_assets_dir / "00000008_000.png").exists() else str(IMAGE_DIR / "00000008_000.png"),
            "labels": "Cardiomegaly",
            "patient_id": 8,
            "age": 69,
            "sex": "F",
        },
        "Effusion (Scan #00000001_002.png)": {
            "filename": "00000001_002.png",
            "path": str(sample_assets_dir / "00000001_002.png") if (sample_assets_dir / "00000001_002.png").exists() else str(IMAGE_DIR / "00000001_002.png"),
            "labels": "Cardiomegaly|Effusion",
            "patient_id": 1,
            "age": 58,
            "sex": "M",
        },
        "Effusion Pure (Scan #00000025_000.png)": {
            "filename": "00000025_000.png",
            "path": str(sample_assets_dir / "00000025_000.png") if (sample_assets_dir / "00000025_000.png").exists() else str(IMAGE_DIR / "00000025_000.png"),
            "labels": "Effusion",
            "patient_id": 25,
            "age": 53,
            "sex": "M",
        },
        "Pneumothorax (Scan #00000013_023.png)": {
            "filename": "00000013_023.png",
            "path": str(sample_assets_dir / "00000013_023.png") if (sample_assets_dir / "00000013_023.png").exists() else str(IMAGE_DIR / "00000013_023.png"),
            "labels": "Infiltration|Mass|Pneumothorax",
            "patient_id": 13,
            "age": 57,
            "sex": "M",
        },
        "Consolidation (Scan #00000032_012.png)": {
            "filename": "00000032_012.png",
            "path": str(sample_assets_dir / "00000032_012.png") if (sample_assets_dir / "00000032_012.png").exists() else str(IMAGE_DIR / "00000032_012.png"),
            "labels": "Atelectasis|Consolidation",
            "patient_id": 32,
            "age": 55,
            "sex": "F",
        },
    }

    # If local metadata and full dataset exist, optionally append extra non-redundant pathology samples
    if IMAGE_DIR.exists() and METADATA_FILE.exists():
        try:
            df = pd.read_csv(METADATA_FILE)
            existing_imgs = set(os.listdir(IMAGE_DIR))
            df = df[df["Image Index"].isin(existing_imgs)].reset_index(drop=True)

            additional_targets = ["Pneumonia", "Infiltration", "Atelectasis"]
            for t in additional_targets:
                match = df[df["Finding Labels"].str.contains(t, case=False, na=False)]
                if not match.empty:
                    row = match.iloc[0]
                    k = f"{t} (Scan #{row['Image Index']})"
                    if k not in samples:
                        samples[k] = {
                            "filename": row["Image Index"],
                            "path": str(IMAGE_DIR / row["Image Index"]),
                            "labels": row["Finding Labels"],
                            "patient_id": row["Patient ID"],
                            "age": row["Patient Age"],
                            "sex": row["Patient Sex"],
                        }
        except Exception:
            pass

    return samples


def main():
    st.markdown('<div class="main-header">🩻 MedicalVision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Chest Radiograph Abnormality Detection, Multi-Pathology Classification & Grad-CAM Localization</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Research Prototype Notice:</strong> MedicalVision AI is developed strictly for research, benchmarking,
            and algorithmic explainability demonstration. It is <strong>not an FDA-cleared diagnostic device</strong> and should never
            replace certified clinical judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        model_scope = st.radio(
            "Diagnostic Model Scope",
            options=["5_classes", "14_classes"],
            format_func=lambda x: "🎯 5 Pathologies Cibles (Haut Signal)" if x == "5_classes" else "🌐 14 Pathologies Globales",
        )
        active_classes = FIVE_DISEASE_CLASSES if model_scope == "5_classes" else DISEASE_CLASSES

        backbone = st.selectbox(
            "Backbone Architecture",
            options=["densenet121", "convnext_tiny", "resnet50", "efficientnet_b0"],
            format_func=lambda x: {
                "densenet121": "DenseNet-121 (CheXNet Standard - 7.0M)",
                "convnext_tiny": "ConvNeXt-Tiny (Modern Pure-CNN - 28.6M)",
                "resnet50": "ResNet-50 (Deep Residual - 23.5M)",
                "efficientnet_b0": "EfficientNet-B0 (Compound Scaling - 4.0M)",
            }.get(x, x),
            disabled=(model_scope == "5_classes"),
        )

        threshold = st.slider("Diagnostic Operating Threshold", min_value=0.10, max_value=0.90, value=DEFAULT_THRESHOLD, step=0.05)
        st.caption("Threshold at which a condition is marked positive.")

        st.subheader("🔍 Explainability Settings")
        cam_alpha = st.slider("Grad-CAM Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
        cam_colormap = st.selectbox("Heatmap Colormap", options=["jet", "turbo", "viridis", "plasma", "inferno"])
        show_bbox = st.checkbox("Show Salient Bounding Box", value=True)

        st.subheader("🎯 Uncertainty Quantification")
        run_uncertainty = st.checkbox("Enable MC Dropout Uncertainty", value=True)
        mc_passes = st.slider("Monte Carlo Forward Passes", min_value=5, max_value=20, value=10, step=1) if run_uncertainty else 0

    # Image Input Selection
    st.markdown("### 🩻 Cliché Radiographique d'Entrée")
    input_source = st.radio(
        "Source du cliché :",
        options=["Cas Clinique Curé (CXR-14)", "Téléverser une Radiographie Personnalisée"],
        horizontal=True,
    )

    active_image = None
    active_filename = "Scan.png"
    meta_info = {"patient_id": "Unknown", "age": "Unknown", "sex": "Unknown", "labels": "Unknown"}

    if input_source == "Cas Clinique Curé (CXR-14)":
        samples = get_curated_samples()
        if samples:
            selected_sample_key = st.selectbox(
                "Sélectionner un cas clinique de référence :",
                options=list(samples.keys()),
                help="Cas cliniques annotés par radiologues certifiés du dataset NIH CXR-14.",
            )
            if selected_sample_key and selected_sample_key in samples:
                info = samples[selected_sample_key]
                img_path = Path(info.get("path", IMAGE_DIR / info["filename"]))
                if img_path.exists():
                    active_image = Image.open(img_path).convert("RGB")
                    active_filename = info["filename"]
                    meta_info = info
        else:
            st.info("Dossier d'images non disponible.")
    else:
        uploaded_file = st.file_uploader(
            "Téléverser un cliché thoracique (PNG, JPG, JPEG) :",
            type=["png", "jpg", "jpeg"],
            help="Images thoraciques en projection frontale (PA ou AP).",
        )
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file).convert("RGB")
            active_filename = uploaded_file.name
            meta_info = {
                "patient_id": "Custom",
                "age": "N/A",
                "sex": "N/A",
                "labels": "Cliché Utilisateur (Non étiqueté)",
            }

    if active_image is None:
        st.warning("Veuillez sélectionner un cas clinique de référence ou téléverser une radiographie.")
        return

    # Image provenance and ground truth banner
    st.info(
        f"🎯 **Cliché Actif pour Diagnostic :** `{active_filename}` | "
        f"**Mode :** {input_source} | "
        f"**Vérité Terrain Clinique :** `{meta_info.get('labels', 'Non étiqueté')}`"
    )

    # Load Model & Run Inference
    if model_scope == "5_classes":
        model = get_cached_5class_model()
    else:
        model = get_cached_model(backbone)

    tensor = preprocess_image_for_inference(active_image)

    with st.spinner("Analyzing thoracic anatomy & calculating multi-label probabilities..."):
        predictions = model.predict_dict(tensor, disease_classes=active_classes)
        uncertainty = None
        if run_uncertainty:
            uncertainty = estimate_uncertainty_mc_dropout(model, tensor, num_samples=mc_passes, disease_classes=active_classes)

    # Top Status Row
    is_abnormal = predictions.get("Is_Abnormal", False)
    abnormal_risk = predictions.get("Abnormal_Risk", 0.0)
    top_disease = max(active_classes, key=lambda d: predictions[d])
    top_prob = predictions[top_disease]

    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Overall Finding",
            value="ABNORMAL" if is_abnormal else "NORMAL",
            delta="Pathology Detected" if is_abnormal else "No Critical Finding",
            delta_color="inverse" if is_abnormal else "normal",
        )
    with m_col2:
        st.metric(label="Thoracic Abnormality Score", value=f"{abnormal_risk:.1%}")
    with m_col3:
        st.metric(label="Primary Suspected Finding", value=top_disease.replace("_", " "), delta=f"{top_prob:.1%}")
    with m_col4:
        uncertainty_status = "Reliable"
        if uncertainty and any(v.get("is_uncertain", False) for v in uncertainty.values()):
            uncertainty_status = "⚠️ Ambiguous"
        st.metric(label="Model Uncertainty", value=uncertainty_status)

    # Detailed Tabs
    tab_explain, tab_diff, tab_detector, tab_probs, tab_uncertainty, tab_report = st.tabs([
        "🔬 Visual Explainability (Grad-CAM)",
        "⚖️ Diagnostic Différentiel Cardio / Effusion",
        "🎯 Supervised Detector (Faster R-CNN)",
        "📊 Multi-Condition Probability Spectrum",
        "⚖️ Model Uncertainty & Calibration",
        "📝 Automated Radiology Report",
    ])

    # Tab 1: Visual Explainability
    with tab_explain:
        st.markdown("### Weakly-Supervised Pathological Localization")
        st.write("Grad-CAM highlights regional pixel gradients that positively support the selected diagnosis.")

        sel_col1, sel_col2 = st.columns([1, 2])
        with sel_col1:
            target_disease = st.selectbox(
                "Select Pathology for Grad-CAM Localization:",
                options=active_classes,
                index=active_classes.index(top_disease),
                format_func=lambda x: f"{x.replace('_', ' ')} ({predictions[x]:.1%})",
            )
            target_idx = active_classes.index(target_disease)

            # Generate Grad-CAM
            gradcam = GradCAM(model)
            orig_w, orig_h = active_image.size
            cam = gradcam.generate(tensor, target_class_idx=target_idx, target_size=(orig_w, orig_h))

            # Overlay
            blended = overlay_heatmap_on_image(active_image, cam, alpha=cam_alpha, colormap_name=cam_colormap)

            # Bounding Box
            boxes = extract_bounding_boxes(cam, threshold=0.5)
            if show_bbox and boxes:
                blended = draw_bounding_boxes_on_image(
                    image=blended,
                    predicted_boxes=boxes,
                    pred_label=f"{target_disease.replace('_', ' ')}",
                )

            st.markdown(f"**Condition:** `{target_disease.replace('_', ' ')}`")
            st.markdown(f"**Predicted Probability:** `{predictions[target_disease]:.1%}`")
            st.markdown(f"**Salient Bounding Boxes Detected:** `{len(boxes)}`")
            for i, b in enumerate(boxes, 1):
                st.caption(f"Box #{i}: [x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}]")

        with sel_col2:
            view_col1, view_col2 = st.columns(2)
            with view_col1:
                st.image(active_image, caption=f"Original Frontal Radiograph ({active_filename})", use_container_width=True)
            with view_col2:
                st.image(blended, caption=f"Grad-CAM Heatmap & Bounding Box ({target_disease})", use_container_width=True)

    # Tab: Diagnostic Différentiel Cardio vs Effusion
    with tab_diff:
        st.markdown("### ⚖️ Diagnostic Différentiel Découplé : Cardiomégalie vs Épanchement Pleural")
        st.write(
            "Ce module analyse la séparation rigoureuse entre la **dilatation de la silhouette cardiaque (Cardiomegaly)** "
            "et l'**épanchement liquidien des culs-de-sac pleuraux (Effusion)** grâce à notre fonction de perte anatomique contrastive."
        )

        p_card = predictions.get("Cardiomegaly", 0.0)
        p_eff = predictions.get("Effusion", 0.0)
        delta_ce = p_card - p_eff

        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.metric("Probabilité Cardiomégalie", f"{p_card:.1%}", delta=f"{'+' if delta_ce >= 0 else ''}{delta_ce:.1%}")
        with d_col2:
            st.metric("Probabilité Épanchement Pleural", f"{p_eff:.1%}", delta=f"{'-' if delta_ce >= 0 else '+'}{abs(delta_ce):.1%}", delta_color="inverse")
        with d_col3:
            if p_card >= 0.50 and p_eff < 0.25:
                verdict = "Cardiomégalie Isolée"
            elif p_eff >= 0.50 and p_card < 0.25:
                verdict = "Épanchement Pleural Isolé"
            elif p_card >= 0.40 and p_eff >= 0.40:
                verdict = "⚠️ Présentation Mixte (ICC)"
            else:
                verdict = "Normal / Non Concluant"
            st.metric("Conclusion Différentielle", verdict)

        if p_card >= 0.50 and p_eff < 0.25:
            st.success("✅ **Cardiomégalie Isolée Détectée** : Élargissement significatif du médiastin / silhouette cardiaque (ICT > 0.5). Les culs-de-sac costo-diaphragmatiques sont libres de liquide.")
        elif p_eff >= 0.50 and p_card < 0.25:
            st.success("✅ **Épanchement Pleural Isolé Détecté** : Émoussement des culs-de-sac pleuraux / opacité basilaire en ménisque. La silhouette cardiaque présente des dimensions normales.")
        elif p_card >= 0.40 and p_eff >= 0.40:
            st.warning("⚠️ **Suspicion d'Insuffisance Cardiaque Congestive (ICC)** : Présence simultanée d'une cardiomégalie et d'un épanchement pleural (transsudat réactionnel). Ce tableau clinique requiert une évaluation hémodynamique et échocardiographique.")
        else:
            st.info("ℹ️ Les indices de cardiomégalie et d'épanchement restent sous les seuils pathologiques critiques.")

        # Dual Grad-CAM Side-by-Side
        if "Cardiomegaly" in active_classes and "Effusion" in active_classes:
            st.markdown("#### 🔍 Focalisation Spatiale Anatomique Comparée (Grad-CAM)")
            c_idx = active_classes.index("Cardiomegaly")
            e_idx = active_classes.index("Effusion")
            gradcam_diff = GradCAM(model)
            orig_w, orig_h = active_image.size
            cam_c = gradcam_diff.generate(tensor, target_class_idx=c_idx, target_size=(orig_w, orig_h))
            cam_e = gradcam_diff.generate(tensor, target_class_idx=e_idx, target_size=(orig_w, orig_h))

            blend_c = overlay_heatmap_on_image(active_image, cam_c, alpha=cam_alpha, colormap_name=cam_colormap)
            blend_e = overlay_heatmap_on_image(active_image, cam_e, alpha=cam_alpha, colormap_name=cam_colormap)

            dc_view1, dc_view2 = st.columns(2)
            with dc_view1:
                st.image(blend_c, caption=f"Focalisation Médiastinale — Cardiomegaly ({p_card:.1%})", use_container_width=True)
            with dc_view2:
                st.image(blend_e, caption=f"Focalisation Basilaire / Pleurale — Effusion ({p_eff:.1%})", use_container_width=True)

    # Tab: Supervised Faster R-CNN Detector
    with tab_detector:
        st.markdown("### Supervised Spatial Localization (Faster R-CNN ResNet-50-FPN)")
        st.write("Trained directly on NIH expert bounding box annotations across 8 focal pathologies.")

        det_col1, det_col2 = st.columns([1, 2])
        detector = get_cached_detector()

        # Resize image for detector (512, 512)
        det_img = active_image.convert("RGB").resize((512, 512))
        img_arr = np.array(det_img).astype(np.float32) / 255.0
        det_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

        detected_boxes = detector.predict(det_tensor)

        with det_col1:
            st.metric("Detected Abnormalities (Supervised)", value=len(detected_boxes))
            if detected_boxes:
                st.markdown("**Predicted Lesions & Confidences:**")
                for i, b in enumerate(detected_boxes, 1):
                    coords = [int(c) for c in b['box']]
                    st.write(f"- **{b['label']}** ({b['score']:.1%}): `[x1={coords[0]}, y1={coords[1]}, x2={coords[2]}, y2={coords[3]}]`")
            else:
                st.info("No bounding boxes detected above operating threshold.")

        with det_col2:
            annotated_img = det_img.copy()
            draw = ImageDraw.Draw(annotated_img)
            color_palette = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
            for idx, b in enumerate(detected_boxes):
                box = b["box"]
                c = color_palette[idx % len(color_palette)]
                draw.rectangle(box, outline=c, width=3)
                draw.text((box[0] + 4, box[1] + 4), f"{b['label']} {b['score']:.1%}", fill=c)

            st.image(annotated_img, caption="Faster R-CNN Supervised Bounding Box Detections", use_container_width=True)

    # Tab 2: Probabilities
    with tab_probs:
        st.markdown("### Multi-Pathology Diagnostic Likelihoods")
        prob_data = pd.DataFrame({
            "Condition": [d.replace("_", " ") for d in active_classes],
            "Probability": [predictions[d] for d in active_classes],
            "Status": ["Positive" if predictions[d] >= threshold else "Negative" for d in active_classes],
        }).sort_values(by="Probability", ascending=True)

        fig = px.bar(
            prob_data,
            x="Probability",
            y="Condition",
            orientation="h",
            color="Status",
            color_discrete_map={"Positive": "#EF4444", "Negative": "#3B82F6"},
            text=prob_data["Probability"].apply(lambda p: f"{p:.1%}"),
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold:.0%})")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Uncertainty
    with tab_uncertainty:
        st.markdown("### Epistemic Model Uncertainty (MC Dropout)")
        st.write("Stochastic forward passes quantify where the network is confident versus ambiguous.")

        if uncertainty:
            uncert_rows = []
            for d in active_classes:
                if d in uncertainty:
                    u = uncertainty[d]
                    uncert_rows.append({
                        "Condition": d.replace("_", " "),
                        "Mean Probability": u["mean_prob"],
                        "Uncertainty (Std Dev σ)": u["std_dev"],
                        "Entropy": u["entropy"],
                        "Flag": "⚠️ High Ambiguity" if u["is_uncertain"] else "✅ Confident",
                    })
            uncert_df = pd.DataFrame(uncert_rows).sort_values(by="Uncertainty (Std Dev σ)", ascending=False)
            st.dataframe(uncert_df, use_container_width=True)

            # Boxplot of samples for top conditions
            top_conditions = sorted(active_classes, key=lambda d: predictions.get(d, 0.0), reverse=True)[:min(5, len(active_classes))]
            fig_box = go.Figure()
            for d in top_conditions:
                if d in uncertainty:
                    fig_box.add_trace(go.Box(y=uncertainty[d]["samples"], name=d.replace("_", " "), boxpoints="all"))
            fig_box.update_layout(
                title=f"Distribution of Predictions Across {mc_passes} MC Dropout Passes (Top Conditions)",
                yaxis_title="Predicted Probability",
                height=400,
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Enable MC Dropout Uncertainty in the sidebar to inspect distribution.")

    # Tab 4: Report
    with tab_report:
        st.markdown("### Automated Radiologist Decision-Support Report")
        report_text = generate_radiology_report(
            image_filename=active_filename,
            predictions=predictions,
            uncertainty=uncertainty,
            threshold=threshold,
            patient_id=str(meta_info.get("patient_id", "Unknown")),
            patient_age=str(meta_info.get("patient_age", "Unknown")),
            patient_sex=str(meta_info.get("patient_sex", "Unknown")),
        )

        st.code(report_text, language="markdown")
        st.download_button(
            label="💾 Download Structured Radiology Report (.txt)",
            data=report_text,
            file_name=f"Radiology_Report_{Path(active_filename).stem}.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
