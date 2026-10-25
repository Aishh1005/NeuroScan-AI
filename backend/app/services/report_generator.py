import json
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)

DISCLAIMER = (
    "AI-assisted research result. This system is not a medical diagnosis and "
    "should not be used as a substitute for professional medical evaluation."
)


def generate_pdf_report(analysis, user, output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], textColor=colors.HexColor("#1e3a8a"))
    section_style = ParagraphStyle("SectionStyle", parent=styles["Heading2"], textColor=colors.HexColor("#1e3a8a"),
                                    spaceBefore=14, spaceAfter=6)
    normal = styles["Normal"]
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], textColor=colors.HexColor("#b91c1c"),
                                       borderColor=colors.HexColor("#b91c1c"), borderWidth=0.75,
                                       borderPadding=8, backColor=colors.HexColor("#fef2f2"))

    story = []
    story.append(Paragraph("Brain MRI AI-Assisted Analysis Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Report ID: {analysis.id}", normal))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", normal))
    story.append(Paragraph(f"Account: {user.full_name} ({user.email})", normal))
    story.append(Spacer(1, 10))
    story.append(Paragraph(DISCLAIMER, disclaimer_style))

    story.append(Paragraph("Upload Details", section_style))
    upload_table_data = [
        ["Image filename", analysis.original_filename],
        ["Analysis date", analysis.created_at.strftime("%Y-%m-%d %H:%M UTC")],
        ["Image quality check", analysis.quality_status.upper()],
    ]
    story.append(_styled_table(upload_table_data))

    story.append(Paragraph("Tumor Detection Result", section_style))
    if analysis.status != "completed":
        story.append(Paragraph(
            f"Analysis did not complete successfully. Reason: {analysis.failure_reason or 'unknown'}",
            normal))
    else:
        result_table = [
            ["Tumor detected", "Yes" if analysis.tumor_detected else "No"],
            ["Predicted class", analysis.predicted_class or "N/A"],
            ["Confidence", f"{(analysis.confidence or 0) * 100:.1f}%"],
            ["Confidence level", (analysis.confidence_level or "N/A").upper()],
            ["Model", f"{analysis.model_name or 'N/A'} ({analysis.model_version or 'N/A'})"],
        ]
        story.append(_styled_table(result_table))

        if analysis.confidence_level == "low":
            story.append(Spacer(1, 6))
            story.append(Paragraph("Low-confidence prediction. Professional review is recommended.",
                                    disclaimer_style))

        if analysis.class_probabilities:
            story.append(Paragraph("Class Probabilities", section_style))
            probs = json.loads(analysis.class_probabilities)
            prob_table = [["Class", "Probability"]] + [
                [k, f"{v * 100:.1f}%"] for k, v in sorted(probs.items(), key=lambda x: -x[1])
            ]
            story.append(_styled_table(prob_table, header=True))

        images_row = []
        if analysis.stored_image_path and os.path.exists(analysis.stored_image_path):
            images_row.append(_labeled_image(analysis.stored_image_path, "Original MRI"))
        if analysis.gradcam_image_path and os.path.exists(analysis.gradcam_image_path):
            images_row.append(_labeled_image(analysis.gradcam_image_path, "Grad-CAM Attention"))
        if analysis.segmentation_image_path and os.path.exists(analysis.segmentation_image_path):
            images_row.append(_labeled_image(analysis.segmentation_image_path, "Tumor Localization"))

        for label, flow in images_row:
            story.append(Paragraph(label, section_style))
            story.append(flow)

        story.append(Paragraph(
            "The highlighted regions in the Grad-CAM visualization represent image areas that "
            "contributed strongly to the model's prediction. This visualization explains model "
            "attention and does not prove the presence or absence of a tumor.",
            normal))

    story.append(Spacer(1, 14))
    story.append(Paragraph(DISCLAIMER, disclaimer_style))

    doc.build(story)
    return output_path


def _styled_table(data, header=False):
    t = Table(data, colWidths=[70 * mm, 90 * mm])
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eff6ff")),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")))
        style.append(("TEXTCOLOR", (0, 0), (-1, 0), colors.white))
    t.setStyle(TableStyle(style))
    return t


def _labeled_image(path, label):
    img = RLImage(path, width=70 * mm, height=70 * mm)
    return (label, img)
