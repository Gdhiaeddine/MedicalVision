"""
Command Line Interface for MedicalVision AI.
Run inference, Grad-CAM visualization, bounding box extraction, and report generation on any Chest X-ray.
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch

from medicalvision.config import DISEASE_CLASSES, DEFAULT_THRESHOLD, OUTPUT_DIR
from medicalvision.models.classifier import build_model
from medicalvision.transforms import preprocess_image_for_inference
from medicalvision.interpretability.gradcam import GradCAM, overlay_heatmap_on_image
from medicalvision.interpretability.localization import extract_bounding_boxes, draw_bounding_boxes_on_image
from medicalvision.uncertainty import estimate_uncertainty_mc_dropout
from medicalvision.report_generator import generate_radiology_report


def parse_args():
    parser = argparse.ArgumentParser(description="MedicalVision AI - Chest X-ray Analysis CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input Chest X-ray image")
    parser.add_argument("--backbone", type=str, default="densenet121", choices=["densenet121", "resnet50"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualization")
    parser.add_argument("--target_disease", type=str, default=None, help="Specific condition for Grad-CAM (e.g. Cardiomegaly)")
    parser.add_argument("--uncertainty", action="store_true", help="Compute MC Dropout epistemic uncertainty")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold for positive prediction")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Directory to save generated artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input X-ray: {image_path.name}")
    raw_image = Image.open(image_path).convert("RGB")
    tensor = preprocess_image_for_inference(raw_image)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {args.backbone} model on {device}...")
    model = build_model(
        backbone=args.backbone,
        pretrained=True if args.checkpoint is None else False,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    model.eval()

    # 1. Multi-label Prediction
    print("\n[Running Diagnostic Inference...]")
    predictions = model.predict_dict(tensor.to(device))

    # 2. Uncertainty (Optional)
    uncertainty_results = None
    if args.uncertainty:
        print("[Estimating Epistemic Uncertainty via MC Dropout (10 passes)...]")
        uncertainty_results = estimate_uncertainty_mc_dropout(model, tensor.to(device))

    # 3. Print Structured Radiology Report
    report = generate_radiology_report(
        image_filename=image_path.name,
        predictions=predictions,
        uncertainty=uncertainty_results,
        threshold=args.threshold,
    )
    print("\n" + report)

    # Save report to output dir
    report_file = output_dir / f"{image_path.stem}_report.txt"
    report_file.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_file}")

    # 4. Grad-CAM & Bounding Box Localization
    if args.gradcam:
        # Determine target disease
        if args.target_disease:
            matching = [i for i, d in enumerate(DISEASE_CLASSES) if d.lower() == args.target_disease.lower().replace(" ", "_")]
            if not matching:
                print(f"Warning: Disease '{args.target_disease}' not recognized. Using top prediction.")
                target_idx = max(range(len(DISEASE_CLASSES)), key=lambda i: predictions[DISEASE_CLASSES[i]])
            else:
                target_idx = matching[0]
        else:
            target_idx = max(range(len(DISEASE_CLASSES)), key=lambda i: predictions[DISEASE_CLASSES[i]])

        target_disease_name = DISEASE_CLASSES[target_idx]
        print(f"\n[Generating Grad-CAM for target condition: {target_disease_name}...]")

        gradcam = GradCAM(model)
        orig_w, orig_h = raw_image.size
        heatmap = gradcam.generate(tensor.to(device), target_class_idx=target_idx, target_size=(orig_w, orig_h))

        # Overlay heatmap
        blended_img = overlay_heatmap_on_image(raw_image, heatmap, alpha=0.45)
        cam_save_path = output_dir / f"{image_path.stem}_gradcam_{target_disease_name}.png"
        blended_img.save(cam_save_path)
        print(f"Grad-CAM overlay saved to: {cam_save_path}")

        # Bounding box extraction
        boxes = extract_bounding_boxes(heatmap, threshold=0.5)
        if boxes:
            bbox_img = draw_bounding_boxes_on_image(
                image=blended_img,
                predicted_boxes=boxes,
                pred_label=f"{target_disease_name} ({predictions[target_disease_name]:.1%})",
            )
            bbox_save_path = output_dir / f"{image_path.stem}_bbox_{target_disease_name}.png"
            bbox_img.save(bbox_save_path)
            print(f"Localized Bounding Box saved to: {bbox_save_path}")


if __name__ == "__main__":
    main()
