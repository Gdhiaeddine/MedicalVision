"""MedicalVision AI - Gradio Interface for Hugging Face Spaces."""
from pathlib import Path
from PIL import Image
import torch
import gradio as gr

from medicalvision.models.classifier import build_model
from medicalvision.config_5class import FIVE_DISEASE_CLASSES
from medicalvision.interpretability.gradcam import GradCAM, overlay_heatmap_on_image
from medicalvision.interpretability.localization import (
    extract_bounding_boxes,
    draw_bounding_boxes_on_image,
)
from medicalvision.transforms import preprocess_image_for_inference

# Paths
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "outputs" / "checkpoints" / "best_model_5class.pth"
SAMPLE_DIR = BASE_DIR / "assets" / "sample_scans"

# Singleton Model Loader
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        has_ckpt = CHECKPOINT_PATH.exists()
        model = build_model(
            backbone="densenet121",
            num_classes=len(FIVE_DISEASE_CLASSES),
            pretrained=False if has_ckpt else True,
            checkpoint_path=str(CHECKPOINT_PATH) if has_ckpt else None,
            device=device,
        )
        model.eval()
        _MODEL = model
    return _MODEL


def analyze_xray(
    image: Image.Image,
    target_disease: str,
    show_bbox: bool,
    cam_alpha: float,
):
    if image is None:
        return None, None, "⚠️ Veuillez charger ou sélectionner un cliché thoracique."

    model = get_model()
    device = next(model.parameters()).device

    # Preprocessing
    rgb_img = image.convert("RGB")
    tensor = preprocess_image_for_inference(rgb_img, device=device)

    # Multi-label inference
    with torch.no_grad():
        preds = model.predict_dict(tensor, disease_classes=FIVE_DISEASE_CLASSES)

    # Probabilities for gr.Label
    probs = {cls_name: float(preds.get(cls_name, 0.0)) for cls_name in FIVE_DISEASE_CLASSES}

    # Grad-CAM heatmap
    if target_disease not in FIVE_DISEASE_CLASSES:
        target_disease = "Cardiomegaly"
    target_idx = FIVE_DISEASE_CLASSES.index(target_disease)

    gradcam = GradCAM(model)
    orig_w, orig_h = rgb_img.size
    cam = gradcam.generate(tensor, target_class_idx=target_idx, target_size=(orig_w, orig_h))

    # Overlay
    blended = overlay_heatmap_on_image(rgb_img, cam, alpha=cam_alpha, colormap_name="jet")

    # Salient Bounding Box
    boxes = extract_bounding_boxes(cam, threshold=0.5)
    if show_bbox and boxes:
        blended = draw_bounding_boxes_on_image(
            image=blended,
            predicted_boxes=boxes,
            pred_label=f"{target_disease} ({preds[target_disease]:.1%})",
        )

    # Cardio vs Effusion Differential
    p_card = probs.get("Cardiomegaly", 0.0)
    p_eff = probs.get("Effusion", 0.0)
    delta_ce = p_card - p_eff

    if delta_ce > 0.15:
        diff_diag = (
            f"### ⚖️ Diagnostic Différentiel Découplé\n\n"
            f"- **Orientation Principale :** <span style='color:#16a34a; font-weight:bold;'>Cardiomégalie Prédominante</span>\n"
            f"- **Score Séparateur :** `Δ(Cardio - Effusion) = +{delta_ce:.1%}`\n"
            f"- **Détail :** Dilatation de la silhouette cardio-médiastinale sans comblement majeur des culs-de-sac pleuraux.\n"
            f"- *Cardiomégalie :* `{p_card:.1%}` | *Épanchement :* `{p_eff:.1%}`"
        )
    elif delta_ce < -0.15:
        diff_diag = (
            f"### ⚖️ Diagnostic Différentiel Découplé\n\n"
            f"- **Orientation Principale :** <span style='color:#2563eb; font-weight:bold;'>Épanchement Pleural Prédominant</span>\n"
            f"- **Score Séparateur :** `Δ(Cardio - Effusion) = {delta_ce:.1%}`\n"
            f"- **Détail :** Opacité basale prédominante avec émoussement du récessus costo-diaphragmatique.\n"
            f"- *Cardiomégalie :* `{p_card:.1%}` | *Épanchement :* `{p_eff:.1%}`"
        )
    elif p_card > 0.35 and p_eff > 0.35:
        diff_diag = (
            f"### ⚖️ Diagnostic Différentiel Découplé\n\n"
            f"- **Orientation Principale :** <span style='color:#d97706; font-weight:bold;'>Comorbidité Cardio-Pleural Mixte</span>\n"
            f"- **Score Séparateur :** `Δ(Cardio - Effusion) = {delta_ce:+.1%}`\n"
            f"- **Détail :** Coexistence suspectée d'une cardiomégalie et d'un épanchement réactionnel (décompensation globale).\n"
            f"- *Cardiomégalie :* `{p_card:.1%}` | *Épanchement :* `{p_eff:.1%}`"
        )
    else:
        diff_diag = (
            f"### ⚖️ Diagnostic Différentiel Découplé\n\n"
            f"- **Orientation :** <span style='color:#6b7280;'>Niveau de risque bas ou équilibré</span>\n"
            f"- **Score Séparateur :** `Δ = {delta_ce:+.1%}`\n"
            f"- *Cardiomégalie :* `{p_card:.1%}` | *Épanchement :* `{p_eff:.1%}`"
        )

    return probs, blended, diff_diag


# Prepare Sample Examples
examples = []
if SAMPLE_DIR.exists():
    for p in sorted(SAMPLE_DIR.glob("*.png")):
        examples.append([str(p), "Cardiomegaly", True, 0.45])

# Build Gradio UI
with gr.Blocks(title="MedicalVision AI - Gradio Space") as demo:
    gr.Markdown(
        """
        # 🩻 MedicalVision AI — Dépistage Thoracique & Grad-CAM
        **Classification Multi-Pathologies Découplée (Cardiomegaly vs Effusion) & Localisation Anatomique Expliquée**
        
        > ⚠️ **Avertissement Prototype de Recherche :** Développé exclusivement pour la recherche et l'explicabilité algorithmique sur le dataset NIH ChestX-ray14. N'est pas un dispositif médical homologué.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Radiographie Thoracique (Frontale PA/AP)")
            target_pathology = gr.Dropdown(
                choices=FIVE_DISEASE_CLASSES,
                value="Cardiomegaly",
                label="Pathologie Cible pour l'Explicabilité (Grad-CAM)",
            )
            with gr.Row():
                show_bbox = gr.Checkbox(value=True, label="Afficher Bounding Box Saliente")
                cam_alpha = gr.Slider(minimum=0.1, maximum=0.9, value=0.45, step=0.05, label="Opacité Grad-CAM")

            analyze_btn = gr.Button("🔍 Lancer l'Analyse Clinique", variant="primary")

        with gr.Column(scale=1):
            output_probs = gr.Label(num_top_classes=5, label="🎯 Probabilités Calibrées des 5 Pathologies Cibles")
            output_cam = gr.Image(type="pil", label="🔬 Localisation Anatomique (Grad-CAM & Bounding Box)")
            output_diff = gr.Markdown()

    analyze_btn.click(
        fn=analyze_xray,
        inputs=[input_image, target_pathology, show_bbox, cam_alpha],
        outputs=[output_probs, output_cam, output_diff],
    )

    if examples:
        gr.Examples(
            examples=examples[:5],
            inputs=[input_image, target_pathology, show_bbox, cam_alpha],
            outputs=[output_probs, output_cam, output_diff],
            fn=analyze_xray,
            cache_examples=False,
            label="📁 Clichés Cliniques de Référence (Cliquer pour tester)",
        )

if __name__ == "__main__":
    demo.launch()
