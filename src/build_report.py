"""Build the concise assignment report from measured run outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUT_PDF = ROOT / "report" / "report.pdf"
NAVY = colors.HexColor("#18324B")
BLUE = colors.HexColor("#2F6B91")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREEN = colors.HexColor("#E7F4EA")
MID_GRAY = colors.HexColor("#5D6975")
LIGHT_GRAY = colors.HexColor("#F1F4F6")
WHITE = colors.white


def format_number(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=21, leading=25, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=MID_GRAY, spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "SectionHeading", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14, leading=17, textColor=NAVY, spaceBefore=4, spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Subheading", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10, leading=13, textColor=BLUE, spaceBefore=5, spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyTextCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.4, leading=12, textColor=colors.HexColor("#253443"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallText", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.1, leading=9.3, textColor=colors.HexColor("#253443"),
        ),
        "tiny": ParagraphStyle(
            "TinyText", parent=base["BodyText"], fontName="Helvetica",
            fontSize=6.5, leading=7.8, textColor=colors.HexColor("#253443"),
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=8.3, leading=11, textColor=NAVY,
        ),
        "center": ParagraphStyle(
            "CenterSmall", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7, leading=8.5, alignment=TA_CENTER,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def styled_table(data, widths, header=True, font_size=7.2, row_padding=4):
    body_cell = ParagraphStyle(
        "TableBodyCell", fontName="Helvetica", fontSize=font_size,
        leading=font_size + 1.6, textColor=colors.HexColor("#253443"),
    )
    header_cell = ParagraphStyle(
        "TableHeaderCell", fontName="Helvetica-Bold", fontSize=font_size,
        leading=font_size + 1.6, textColor=WHITE,
    )
    converted = []
    for row_index, row in enumerate(data):
        cell_style = header_cell if header and row_index == 0 else body_cell
        converted.append([
            Paragraph(value, cell_style) if isinstance(value, str) and value else value
            for value in row
        ])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), row_padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), row_padding),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE5")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#253443")),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
        for row in range(1, len(data)):
            if row % 2 == 0:
                commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(40, letter[1] - 26, "BT2024146  |  MACHINE LEARNING ASSIGNMENT 1")
    canvas.setStrokeColor(colors.HexColor("#CDD8E0"))
    canvas.setLineWidth(0.5)
    canvas.line(40, letter[1] - 33, letter[0] - 40, letter[1] - 33)
    canvas.setFillColor(MID_GRAY)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(40, 23, "Polynomial regression | Test labels were not available")
    canvas.drawRightString(letter[0] - 40, 23, f"Page {doc.page}")
    canvas.restoreState()


def load_run_data():
    summary = json.loads((OUTPUTS / "selection_summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((OUTPUTS / "data_diagnostics.json").read_text(encoding="utf-8"))
    selections = summary["problems"]
    cv = {
        v: pd.read_csv(OUTPUTS / f"model_selection_{v}.csv")
        for v in ("var1", "var2")
    }
    return summary, diagnostics, selections, cv


def best_ridge_by_degree(results: pd.DataFrame) -> pd.DataFrame:
    ridge = results[results["family"] == "Ridge"].copy()
    ridge = ridge.sort_values("mean_validation_mse")
    return ridge.groupby("degree", as_index=False).first().sort_values("degree")


def build() -> Path:
    summary, diagnostics, selections, cv = load_run_data()
    style = build_styles()
    report_dir = OUT_PDF.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=47,
        bottomMargin=40,
        title="BT2024146 Polynomial Regression Assignment Report",
        author="BT2024146",
        subject="Polynomial regression model selection for var1 and var2",
    )
    story = []

    # Page 1: the assigned problems and observed data.
    story.append(para("Polynomial Regression", style["title"]))
    story.append(para("Machine Learning Assignment 1 | BT2024146 | 8 October 2026", style["subtitle"]))
    story.append(
        para(
            "This report describes two separate polynomial-regression problems using the assigned data. "
            "The official assignment sets maximum total degrees of 10 for var1 and 20 for var2, and evaluates "
            "predictions against hidden test labels using mean squared error (MSE) and R-squared. The test labels "
            "were not available for this work.",
            style["body"],
        )
    )
    story.append(para("Problems and data", style["h1"]))
    overview = [
        ["Problem", "Input features", "Target", "Train / test rows", "Max degree"],
        ["var1", "6: x1 to x6", "Net Power Score (y)", "1,000 / 1,000", "10"],
        ["var2", "3: x1 to x3", "Thermal Anomaly Score (y)", "1,000 / 1,000", "20"],
    ]
    story.append(styled_table(overview, [52, 100, 160, 110, 78], row_padding=5))
    story.append(Spacer(1, 8))
    story.append(para("Feature descriptions from the assignment", style["h2"]))
    feature_data = [
        ["var1: power-plant controls", "var2: survey coordinates"],
        [
            "x1 high-pressure steam valve; x2 condenser coolant flow; x3 reinjection pump pressure; "
            "x4 turbine blade pitch; x5 non-condensable gas exhaust valve; x6 steam inlet pressure.",
            "x1 east-west offset; x2 north-south offset; x3 vertical depth offset from the basecamp.",
        ],
    ]
    story.append(styled_table(feature_data, [265, 265], row_padding=6, font_size=7.1))
    story.append(Spacer(1, 8))
    story.append(para("Observed data checks", style["h2"]))
    dq = [["Problem", "Feature range", "Target range (mean; SD)", "Missing", "Duplicate rows: train / test"]]
    for diagnostic in diagnostics:
        v = diagnostic["variable"]
        target = diagnostic["target_summary"]
        lo = target["min"]
        hi = target["max"]
        mean = target["mean"]
        std = target["std"]
        dups = f"{diagnostic['train_duplicate_rows']} / {diagnostic['test_duplicate_feature_rows']}"
        dq.append(
            [
                v,
                "[-1, 1]",
                para(f"{lo:.2f} to {hi:.2f}<br/>({mean:.2f}; {std:.2f})", style["small"]),
                "0",
                dups,
            ]
        )
    story.append(styled_table(dq, [52, 85, 165, 70, 158], row_padding=5, font_size=7.1))
    story.append(Spacer(1, 5))
    story.append(
        para(
            "All supplied training and test feature columns match within each problem; no missing or non-finite "
            "values were found. Test feature duplicates (2 in var1 and 6 in var2) were retained in their original "
            "positions. The sample submission has 1,000 rows and one column named y.",
            style["small"],
        )
    )

    # Page 2: validation and model-selection design.
    story.append(PageBreak())
    story.append(para("Validation and model search", style["h1"]))
    story.append(
        para(
            "Each problem has 1,000 labeled rows and no time or group identifier requiring a structured split. "
            "The comparison therefore uses RepeatedKFold with five shuffled folds, repeated twice, and random seed "
            "2026. Each fold trains on 800 rows and validates on 200; results are averaged over 10 validation "
            "folds. Repeating the shuffled partition reduces dependence on a single split while keeping the "
            "degree search computationally manageable.",
            style["body"],
        )
    )
    cv_box = Table(
        [[
            para("5 folds x 2 repeats", style["callout"]),
            para("800 train / 200 validation per fold", style["callout"]),
            para("Shuffle = yes; seed = 2026", style["callout"]),
        ]],
        colWidths=[170, 180, 170],
    )
    cv_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#B8CAD8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(cv_box)
    story.append(Spacer(1, 9))
    story.append(para("Polynomial models", style["h2"]))
    story.append(
        para(
            "PolynomialFeatures expands all monomials whose total degree is at most the candidate degree. "
            "The model search covers var1 degrees 1-10 and var2 degrees 1-20. With six inputs, degree 10 has "
            "8,007 non-constant terms; with three inputs, degree 20 has 1,770. These sizes make unregularized "
            "high-degree fits unstable or underdetermined, so Ridge regularization was evaluated at every degree.",
            style["body"],
        )
    )
    candidate_table = [
        ["Candidate", "Search", "Purpose"],
        [
            "Ridge polynomial regression",
            "All allowed degrees; alpha in 0.01, 0.1, 1, 10, 100, 1,000, 10,000, 100,000",
            "Controls coefficient size and high-degree variance.",
        ],
        [
            "Unregularized polynomial least squares",
            "var1 degrees 1-5; var2 degrees 1-14",
            "Baseline where the feature count is below 800 training rows.",
        ],
    ]
    story.append(styled_table(candidate_table, [125, 245, 150], row_padding=5, font_size=7.1))
    story.append(Spacer(1, 7))
    story.append(para("Leakage control and selection", style["h2"]))
    story.append(
        para(
            "Within each validation fold, the polynomial map is deterministic and the StandardScaler is fitted "
            "only on that fold's 800 training rows, then applied to its 200 validation rows. The validation code "
            "solves the same Ridge objective in dual form to reuse a fold's matrix factorization across alpha "
            "values; the final selected model is refit with a scikit-learn pipeline on all 1,000 labeled rows. "
            "MSE and R-squared are averaged over folds. The one-standard-error rule takes the lowest degree with "
            "mean validation MSE no more than one fold-to-fold standard error above the minimum; within that "
            "degree it takes the lowest-MSE candidate.",
            style["body"],
        )
    )

    # Page 3: scores and degree sweeps.
    story.append(PageBreak())
    story.append(para("Model-selection results", style["h1"]))
    selected_table = [["Problem", "Selected model", "Terms", "CV MSE", "CV R-squared", "Min CV MSE"]]
    for v in ("var1", "var2"):
        chosen = selections[v]["selected"]
        selected_table.append(
            [
                v,
                f"Ridge, degree {chosen['degree']}, alpha {chosen['alpha']:g}",
                f"{chosen['polynomial_terms']:,}",
                format_number(chosen["mean_validation_mse"], 6),
                format_number(chosen["mean_validation_r2"], 6),
                format_number(chosen["minimum_mean_validation_mse"], 6),
            ]
        )
    story.append(styled_table(selected_table, [50, 142, 58, 84, 94, 94], row_padding=5, font_size=7.1))
    story.append(Spacer(1, 5))
    story.append(
        para(
            "Best Ridge alpha and validation scores at each degree; var1 is on the left and var2 on the right. "
            "MSE is mean squared error; R2 is R-squared. Green cells mark the selected degrees.",
            style["small"],
        )
    )
    ridge1 = best_ridge_by_degree(cv["var1"])
    ridge2 = best_ridge_by_degree(cv["var2"])
    degree_data = [["d1", "alpha", "MSE", "R2", "", "d2", "alpha", "MSE", "R2"]]
    map1 = {int(row.degree): row for row in ridge1.itertuples(index=False)}
    map2 = {int(row.degree): row for row in ridge2.itertuples(index=False)}
    for degree in range(1, 21):
        cells = []
        for mapping, max_degree in ((map1, 10), (map2, 20)):
            row = mapping.get(degree) if degree <= max_degree else None
            if row is None:
                cells.extend(["", "", "", ""])
            else:
                cells.extend([
                    str(degree),
                    f"{float(row.alpha):g}",
                    format_number(row.mean_validation_mse, 4),
                    format_number(row.mean_validation_r2, 4),
                ])
        degree_data.append(cells[:4] + [""] + cells[4:])
    degree_table = styled_table(
        degree_data,
        [36, 54, 72, 75, 10, 36, 54, 72, 75],
        header=True,
        font_size=6.6,
        row_padding=2.25,
    )
    chosen_degree1 = int(selections["var1"]["selected"]["degree"])
    chosen_degree2 = int(selections["var2"]["selected"]["degree"])
    degree_table.setStyle(TableStyle([
        ("BACKGROUND", (0, chosen_degree1), (3, chosen_degree1), PALE_GREEN),
        ("FONTNAME", (0, chosen_degree1), (3, chosen_degree1), "Helvetica-Bold"),
        ("BACKGROUND", (5, chosen_degree2), (8, chosen_degree2), PALE_GREEN),
        ("FONTNAME", (5, chosen_degree2), (8, chosen_degree2), "Helvetica-Bold"),
    ]))
    story.append(Spacer(1, 6))
    story.append(degree_table)
    story.append(Spacer(1, 7))
    ols_rows = [["Problem", "Best OLS degree", "Validation MSE", "Validation R-squared"]]
    for v in ("var1", "var2"):
        best_ols = cv[v][cv[v]["family"] == "OLS"].sort_values("mean_validation_mse").iloc[0]
        ols_rows.append([
            v,
            str(int(best_ols["degree"])),
            format_number(best_ols["mean_validation_mse"], 6),
            format_number(best_ols["mean_validation_r2"], 6),
        ])
    story.append(styled_table(ols_rows, [90, 120, 150, 170], row_padding=4, font_size=7))
    story.append(Spacer(1, 4))
    story.append(
        para(
            "Ridge had lower validation MSE than the best tested OLS candidate in both problems. The selected "
            "Ridge settings are recorded with fold scores and training scores in outputs/model_selection_var1.csv "
            "and outputs/model_selection_var2.csv.",
            style["small"],
        )
    )

    # Page 4: rationale, limitations, and output verification.
    story.append(PageBreak())
    story.append(para("Selection rationale and final predictions", style["h1"]))
    var1 = selections["var1"]["selected"]
    var2 = selections["var2"]["selected"]
    best2 = cv["var2"].sort_values("mean_validation_mse").iloc[0]
    best_var1_degree_10 = cv["var1"][(cv["var1"]["family"] == "Ridge") & (cv["var1"]["degree"] == 10)].sort_values("mean_validation_mse").iloc[0]
    best_var2_degree_15 = cv["var2"][(cv["var2"]["family"] == "Ridge") & (cv["var2"]["degree"] == 15)].sort_values("mean_validation_mse").iloc[0]
    story.append(para("Why these models were selected", style["h2"]))
    story.append(
        para(
            f"<b>var1:</b> Degree 5 with Ridge alpha 10 achieved the lowest mean validation MSE "
            f"({var1['mean_validation_mse']:.6f}; R-squared {var1['mean_validation_r2']:.6f}) among all tested "
            f"settings. Its 461 monomials keep the model below the 800-row fold size. At degree 10, the best "
            f"Ridge model's training MSE was lower ({float(best_var1_degree_10['mean_training_mse']):.3f} versus "
            f"{var1['mean_training_mse']:.3f} at degree 5), while validation MSE was higher "
            f"({float(best_var1_degree_10['mean_validation_mse']):.3f} versus {var1['mean_validation_mse']:.3f}), "
            "consistent with higher variance at the maximum degree.",
            style["body"],
        )
    )
    story.append(
        para(
            f"<b>var2:</b> Degree 11 with alpha 1 had the minimum mean validation MSE "
            f"({float(best2['mean_validation_mse']):.6f}; R-squared {float(best2['mean_validation_r2']):.6f}). "
            f"Degree 10 with alpha 1 scored {var2['mean_validation_mse']:.6f}, below the one-standard-error "
            f"cutoff of {var2['one_se_cutoff_mse']:.6f}, and uses 285 terms rather than degree 11's "
            f"{int(best2['polynomial_terms'])} terms. From degree 11 to 15, the best-per-degree training MSE "
            f"fell from {float(best2['mean_training_mse']):.3f} to "
            f"{float(best_var2_degree_15['mean_training_mse']):.3f}, while validation MSE rose from "
            f"{float(best2['mean_validation_mse']):.3f} to "
            f"{float(best_var2_degree_15['mean_validation_mse']):.3f}; the lower degree was retained for the "
            "generalization tradeoff.",
            style["body"],
        )
    )
    train_val = [["Problem", "Mean training MSE", "Mean validation MSE", "Mean training R-squared", "Mean validation R-squared"]]
    for v, chosen in (("var1", var1), ("var2", var2)):
        train_val.append([
            v,
            format_number(chosen["mean_training_mse"], 4),
            format_number(chosen["mean_validation_mse"], 4),
            format_number(chosen["mean_training_r2"], 4),
            format_number(chosen["mean_validation_r2"], 4),
        ])
    story.append(styled_table(train_val, [62, 112, 112, 123, 121], row_padding=5, font_size=6.9))
    story.append(Spacer(1, 7))
    story.append(para("Final prediction procedure", style["h2"]))
    story.append(
        para(
            "After selection, each pipeline was refit on all 1,000 labeled rows for its problem and applied to "
            "the corresponding 1,000 test feature rows without sorting. The written files were reopened and checked "
            "automatically: each has exactly one y column, 1,000 finite predictions, and no index column. The two "
            "outputs are outputs/BT2024146_pred_var1.csv and outputs/BT2024146_pred_var2.csv. No hidden-test score "
            "is reported.",
            style["body"],
        )
    )
    story.append(para("Reproduction", style["h2"]))
    story.append(
        para(
            "From the project root, install requirements.txt and run "
            "<font face='Courier'>python src/train_and_predict.py</font>. This recreates the CV tables, diagnostics, "
            "selection summary, and both prediction files. Run "
            "<font face='Courier'>python src/build_report.py</font> to regenerate this report. The project and "
            "assigned data copies are intended for the provided GitHub repository: "
            "<link href='https://github.com/Varun576253/ML_Assignment' color='#2F6B91'>github.com/Varun576253/ML_Assignment</link>.",
            style["small"],
        )
    )

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return OUT_PDF


if __name__ == "__main__":
    print(build())
