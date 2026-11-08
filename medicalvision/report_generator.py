"""
Automated Structured Radiology Report Generator for MedicalVision AI.
Transforms model predictions, Grad-CAM findings, and uncertainty metrics
into a structured clinical decision support summary.
"""

from datetime import datetime
from typing import Dict, List, Optional
from medicalvision.config import DISEASE_CLASSES, DEFAULT_THRESHOLD


# Anatomical mapping for grouping findings in structured reports
ANATOMICAL_ZONES = {
    "Lungs & Parenchyma": ["Atelectasis", "Infiltration", "Mass", "Nodule", "Pneumonia", "Consolidation", "Emphysema", "Fibrosis"],
    "Pleura & Diaphragm": ["Effusion", "Pneumothorax", "Pleural_Thickening", "Hernia"],
    "Cardiac & Vascular": ["Cardiomegaly", "Edema"],
}


def generate_radiology_report(
    image_filename: str,
    predictions: Dict[str, float],
    uncertainty: Optional[Dict[str, Dict[str, float]]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    patient_id: Optional[str] = "N/A",
    patient_age: Optional[str] = "N/A",
    patient_sex: Optional[str] = "N/A",
) -> str:
    """
    Format a standardized radiologist decision-support report in Markdown format.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Detect positive findings
    positive_findings = []
    uncertain_findings = []

    target_diseases = [k for k in predictions.keys() if k not in ("Is_Abnormal", "Abnormal_Risk")]

    for disease in target_diseases:
        prob = predictions.get(disease, 0.0)
        is_pos = prob >= threshold
        is_uncert = False
        uncert_val = 0.0

        if uncertainty and disease in uncertainty:
            is_uncert = uncertainty[disease].get("is_uncertain", False)
            uncert_val = uncertainty[disease].get("std_dev", 0.0)

        if is_pos:
            positive_findings.append((disease, prob, uncert_val, is_uncert))
        elif is_uncert:
            uncertain_findings.append((disease, prob, uncert_val))

    # Sort positives by descending probability
    positive_findings.sort(key=lambda x: x[1], reverse=True)

    # Build report text
    report = []
    report.append("=" * 60)
    report.append("           MEDICALVISION AI - RADIOLOGY DECISION SUPPORT")
    report.append("           ** RESEARCH PROTOTYPE - NOT FOR DIAGNOSTIC USE **")
    report.append("=" * 60)
    report.append(f"Study Image:     {image_filename}")
    report.append(f"Analysis Time:   {timestamp}")
    report.append(f"Patient ID:      {patient_id} | Age: {patient_age} | Sex: {patient_sex}")
    report.append(f"View Position:   PA / AP Frontal Projection")
    report.append("-" * 60)

    # Clinical Impression
    report.append("\n[IMPRESSION]")
    disease_probs = [float(v) for k, v in predictions.items() if k not in ("Is_Abnormal", "Abnormal_Risk") and isinstance(v, (int, float))]
    highest_non_sig = max(disease_probs) if disease_probs else 0.0

    if not positive_findings:
        report.append("  * NO CRITICAL THORACIC ABNORMALITY DETECTED (Screening Normal).")
        report.append(f"  * Highest non-significant probability: {highest_non_sig:.1%}")
    else:
        report.append(f"  * ABNORMAL EXAMINATION - {len(positive_findings)} Finding(s) Exceeding Operating Threshold ({threshold:.0%}):")
        for disease, prob, uncert_val, is_uncert in positive_findings:
            d_name = disease.replace("_", " ")
            uncert_str = f" [Uncertainty σ={uncert_val:.2f}]" if uncert_val > 0 else ""
            alert_str = " ⚠️ Second Reader Recommended" if is_uncert else ""
            report.append(f"    - {d_name:<20}: Probability {prob:>6.1%}{uncert_str}{alert_str}")

    # Regional Findings
    report.append("\n[DETAILED REGIONAL FINDINGS]")
    for zone, diseases in ANATOMICAL_ZONES.items():
        zone_diseases = [d for d in diseases if d in target_diseases]
        if not zone_diseases:
            continue
        report.append(f"\n  -- {zone} --")
        for d in zone_diseases:
            prob = predictions.get(d, 0.0)
            status = "POSITIVE" if prob >= threshold else "Negative"
            d_name = d.replace("_", " ")
            report.append(f"    * {d_name:<20}: {prob:>5.1%}  [{status}]")

    # Uncertainty / Quality Assessment
    report.append("\n[RELIABILITY & QUALITY CONTROL]")
    if uncertain_findings:
        report.append("  * ⚠️ Ambiguous cases detected (elevated epistemic variance):")
        for d, prob, uncert in uncertain_findings:
            d_name = d.replace("_", " ")
            report.append(f"    - {d_name}: prob={prob:.1%}, σ={uncert:.2f}")
        report.append("  * Recommendation: Clinical correlation and expert radiologist review advised.")
    else:
        report.append("  * Model prediction certainty within high confidence boundaries.")

    report.append("\n" + "=" * 60)
    return "\n".join(report)
