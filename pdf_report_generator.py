"""
PDF Performance Report Generator Module
Student Performance Prediction and Analysis System
Generates professional academic assessment PDF reports for college presentation.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_student_performance_pdf(
    student_name: str,
    predicted_category: str,
    risk_tier: str,
    risk_color_hex: str,
    input_values: Dict[str, Any],
    probabilities: Dict[str, float],
    top_features: List[Dict[str, Any]],
    recommendations: List[str],
    cohort_benchmarks: Optional[Dict[str, Any]] = None,
    model_metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate a publication-grade academic assessment PDF report.
    Returns PDF binary bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0B132B"),
        alignment=0,
    )

    sub_title_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=0,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=6,
    )

    cell_style = ParagraphStyle(
        "CellNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )

    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
    )

    rec_style = ParagraphStyle(
        "RecStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
    )

    elements = []

    # -------------------------------------------------------------
    # 1. HEADER & META
    # -------------------------------------------------------------
    elements.append(Paragraph("Student Performance Prediction and Analysis System", title_style))
    elements.append(Paragraph("Official Academic Performance Diagnostic & Risk Assessment Report", sub_title_style))
    elements.append(Spacer(1, 6))

    now_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    meta_table_data = [
        [
            Paragraph(f"<b>Student / Assessor:</b> {student_name}", cell_style),
            Paragraph(f"<b>Assessment Date:</b> {now_str}", cell_style),
            Paragraph(f"<b>Algorithm:</b> Decision Tree Classifier", cell_style),
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[2.3 * inch, 2.7 * inch, 2.3 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 2. EXECUTIVE CLASSIFICATION BANNER
    # -------------------------------------------------------------
    elements.append(Paragraph("Executive Performance Classification", section_heading))

    cat_color = colors.HexColor(risk_color_hex) if risk_color_hex else colors.HexColor("#2563EB")
    exec_data = [
        [
            Paragraph("<b>Predicted Category:</b>", cell_style),
            Paragraph(f"<b><font size=12 color='{risk_color_hex}'>{predicted_category.upper()}</font></b>", cell_bold),
            Paragraph("<b>Academic Risk Tier:</b>", cell_style),
            Paragraph(f"<b>{risk_tier}</b>", cell_bold),
        ]
    ]

    exec_table = Table(exec_data, colWidths=[1.8 * inch, 1.9 * inch, 1.8 * inch, 1.8 * inch])
    exec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, cat_color),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(exec_table)
    elements.append(Spacer(1, 8))

    # Probabilities row
    if probabilities:
        prob_strs = [f"{cls}: <b>{p:.1f}%</b>" for cls, p in probabilities.items()]
        prob_summary = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(prob_strs)
        elements.append(Paragraph(f"<b>Multiclass Leaf Probabilities:</b> {prob_summary}", cell_style))
        elements.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # 3. STUDENT INPUT EVALUATION ATTRIBUTES
    # -------------------------------------------------------------
    elements.append(Paragraph("Evaluated Student Attribute Inputs", section_heading))

    input_rows = [["Predictor Attribute", "Assessed Value", "Cohort Benchmark Mean", "Standing"]]

    for feat, val in input_values.items():
        feat_label = str(feat).replace("_", " ").title()
        val_str = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)

        # Benchmark comparison if available
        bench_mean_str = "-"
        standing_str = "-"
        if cohort_benchmarks and feat in cohort_benchmarks:
            b_mean = cohort_benchmarks[feat].get("mean")
            if b_mean is not None and isinstance(val, (int, float)):
                bench_mean_str = f"{b_mean:.1f}"
                if val >= b_mean:
                    standing_str = "Above Average"
                else:
                    standing_str = "Below Average"

        input_rows.append([feat_label, val_str, bench_mean_str, standing_str])

    input_table = Table(input_rows, colWidths=[2.6 * inch, 1.6 * inch, 1.6 * inch, 1.5 * inch])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("PADDING", (0, 0), (-1, -1), 4.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    elements.append(input_table)
    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 4. DECISION TREE INFLUENCE FACTORS
    # -------------------------------------------------------------
    if top_features:
        elements.append(Paragraph("Decision Tree Primary Influencing Features", section_heading))
        feat_rows = [["Rank", "Feature Variable", "Model Importance", "Relative Contribution (%)"]]
        for i, item in enumerate(top_features[:4], start=1):
            fname = item.get("Feature", "").replace("_", " ").title()
            imp = item.get("Importance", 0.0)
            pct = item.get("Percentage", 0.0)
            feat_rows.append([str(i), fname, f"{imp:.4f}", f"{pct:.2f}%"])

        feat_table = Table(feat_rows, colWidths=[0.8 * inch, 2.7 * inch, 1.9 * inch, 1.9 * inch])
        feat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8.5),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        elements.append(feat_table)
        elements.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 5. PERSONALIZED ACADEMIC RECOMMENDATIONS
    # -------------------------------------------------------------
    elements.append(Paragraph("Personalized Academic Recommendations", section_heading))
    if recommendations:
        for idx, rec in enumerate(recommendations, start=1):
            bullet_text = f"<b>{idx}.</b> {rec}"
            elements.append(Paragraph(bullet_text, rec_style))
            elements.append(Spacer(1, 3))
    else:
        elements.append(Paragraph("Maintain existing academic routine and consistency.", rec_style))

    elements.append(Spacer(1, 14))

    # -------------------------------------------------------------
    # 6. SIGN-OFF & DISCLAIMER
    # -------------------------------------------------------------
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    elements.append(Paragraph(
        "<b>Educational Advisory Disclaimer:</b> This performance diagnostic report was computed using an empirical "
        "Scikit-Learn Decision Tree Classifier and rule-based academic policies. It is intended for student guidance, "
        "curriculum counseling, and semester intervention planning. It does not replace formal collegiate grading or institutional transcripts.",
        disclaimer_style
    ))

    # Build document
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

