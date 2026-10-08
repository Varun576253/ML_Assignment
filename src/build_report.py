"""Build a concise report from the final model-selection outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUT_PDF = ROOT / "report" / "report.pdf"
NAVY = colors.HexColor("#18324B")
BLUE = colors.HexColor("#2F6B91")
PALE_BLUE = colors.HexColor("#EAF2F8")
LIGHT_GRAY = colors.HexColor("#F1F4F6")
WHITE = colors.white
PROBLEMS = ("var1", "var2")
TARGET_NAMES = {"var1": "Net Power Score", "var2": "Thermal Anomaly Score"}


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=21, leading=25, textColor=NAVY, alignment=0, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=colors.HexColor("#5D6975"), spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "SectionHeading", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, textColor=NAVY, spaceBefore=5, spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Subheading", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=9.5, leading=12, textColor=BLUE, spaceBefore=6, spaceAfter=3,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.2, leading=11, textColor=colors.HexColor("#253443"),
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "ReportSmall", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.2, leading=9.2, textColor=colors.HexColor("#253443"),
            spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "TableBody", fontName="Helvetica", fontSize=7.1, leading=8.6,
            textColor=colors.HexColor("#253443"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader", fontName="Helvetica-Bold", fontSize=7.1, leading=8.6,
            textColor=WHITE,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=8.2, leading=10.5, textColor=NAVY,
        ),
    }


def make_table(rows, widths, styles, font_size=7.1, padding=4):
    body_style = ParagraphStyle(
        "SizedTableBody", parent=styles["table"],
        fontSize=font_size, leading=font_size + 1.5,
    )
    header_style = ParagraphStyle(
        "SizedTableHeader", parent=styles["table_header"],
        fontSize=font_size, leading=font_size + 1.5,
    )
    converted = []
    for row_index, row in enumerate(rows):
        cell_style = header_style if row_index == 0 else body_style
        converted.append([
            value if isinstance(value, Paragraph) else Paragraph(str(value), cell_style)
            for value in row
        ])
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE5")),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ]
    for row_index in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def decorate_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#5D6975"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(40, 23, "Polynomial regression | Hidden test labels unavailable")
    canvas.drawRightString(letter[0] - 40, 23, f"Page {doc.page}")
    canvas.restoreState()


def load_outputs():
    summary = json.loads((OUTPUTS / "selection_summary.json").read_text(encoding="utf-8"))
    diagnostics = {
        item["variable"]: item
        for item in json.loads((OUTPUTS / "data_diagnostics.json").read_text(encoding="utf-8"))
    }
    cv = {name: pd.read_csv(OUTPUTS / f"model_selection_{name}.csv") for name in PROBLEMS}
    return summary["problems"], diagnostics, cv


def best_ridge(results: pd.DataFrame, degree: int) -> pd.Series:
    subset = results[(results["family"] == "Ridge") & (results["degree"] == degree)]
    return subset.sort_values("mean_validation_mse").iloc[0]


def build() -> Path:
    selections, diagnostics, cv = load_outputs()
    styles = make_styles()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=letter, leftMargin=40, rightMargin=40,
        topMargin=47, bottomMargin=40,
        title="BT2024146 Polynomial Regression Assignment Report",
        author="BT2024146",
        subject="Polynomial regression model selection for var1 and var2",
    )
    story = []

    # Page 1: problem setup, data checks, and the validation method.
    story.extend([
        Paragraph("Polynomial Regression", styles["title"]),
        Paragraph("Machine Learning Assignment 1 | Roll No. BT2024146", styles["subtitle"]),
        Paragraph(
            "Two separate regression tasks predict the assigned target variables from their input features. "
            "Each task has 1,000 training rows and 1,000 test rows. The assignment limits total polynomial "
            "degree to 10 for var1 and 20 for var2; hidden test labels were unavailable.",
            styles["body"],
        ),
        Paragraph("Problems and data", styles["h1"]),
    ])

    overview = [["Problem", "Features", "Target", "Train / test", "Target range", "Max degree"]]
    for name in PROBLEMS:
        item = diagnostics[name]
        target = item["target_summary"]
        overview.append([
            name,
            f"{len(item['feature_columns'])}: " + ", ".join(item["feature_columns"]),
            TARGET_NAMES[name],
            f"{item['train_shape'][0]:,} / {item['test_shape'][0]:,}",
            f"{target['min']:.2f} to {target['max']:.2f}",
            str(selections[name]["max_allowed_degree"]),
        ])
    story.append(make_table(overview, [48, 100, 145, 76, 103, 60], styles, padding=5))
    feature_ranges = [
        value
        for item in diagnostics.values()
        for value in item["feature_ranges_train"].values()
    ]
    feature_min = min(value[0] for value in feature_ranges)
    feature_max = max(value[1] for value in feature_ranges)
    train_dups = sum(item["train_duplicate_rows"] for item in diagnostics.values())
    test_dups = [diagnostics[name]["test_duplicate_feature_rows"] for name in PROBLEMS]
    missing = sum(sum(item["train_missing_by_column"].values()) + sum(item["test_missing_by_column"].values()) for item in diagnostics.values())
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f"Observed checks: feature range {feature_min:.0f} to {feature_max:.0f}; {missing} missing values; "
        f"{train_dups} duplicate training rows; test feature duplicates: {test_dups[0]} (var1), "
        f"{test_dups[1]} (var2). Duplicate test rows were kept in their supplied order.",
        styles["small"],
    ))

    story.extend([
        Paragraph("Validation and method", styles["h1"]),
        Paragraph(
            "Model comparisons use shuffled RepeatedKFold cross-validation with 5 folds and 2 repeats "
            "(10 validation folds total; random seed 2026). Each fold trains on 800 rows and validates on 200. "
            "The repeated splits reduce dependence on one random partition while keeping the full permitted "
            "degree searches practical.",
            styles["body"],
        ),
        Paragraph(
            "PolynomialFeatures expands all monomials up to each candidate total degree. Ridge regression "
            "searched degrees 1-10 for var1 and 1-20 for var2, with the configured alpha grid at every degree. "
            "Unregularized least squares "
            "was also evaluated where the expanded feature count was below 800. The one-standard-error rule "
            "selects the lowest degree within one fold-to-fold standard error of the minimum mean validation MSE.",
            styles["body"],
        ),
        Paragraph(
            "Leakage control: polynomial expansion is deterministic, and StandardScaler is fitted only on each "
            "fold's training rows before transforming that fold's validation rows. After selection, each final "
            "pipeline is refit on all 1,000 labeled rows. MSE and R-squared are averaged over validation folds.",
            styles["body"],
        ),
    ])

    # Page 2: selected models, the one-SE decision, and prediction/reproduction details.
    story.append(PageBreak())
    story.append(Paragraph("Results and final predictions", styles["h1"]))
    model_rows = [["Problem", "Selected polynomial Ridge", "Terms", "CV MSE", "CV R-squared"]]
    for name in PROBLEMS:
        selected = selections[name]["selected"]
        model_rows.append([
            name,
            f"Degree {selected['degree']}, alpha {selected['alpha']:g}",
            f"{selected['polynomial_terms']:,}",
            f"{selected['mean_validation_mse']:.6f}",
            f"{selected['mean_validation_r2']:.6f}",
        ])
    story.append(make_table(model_rows, [55, 190, 65, 95, 110], styles, padding=5))
    story.append(Spacer(1, 6))

    selected2 = selections["var2"]["selected"]
    raw_min2 = cv["var2"].loc[cv["var2"]["mean_validation_mse"].idxmin()]
    if int(raw_min2["degree"]) == 11 and int(selected2["degree"]) == 10:
        var2_selection_text = (
            "Degree 11 achieved the lowest raw cross-validation MSE. However, degree 10 was selected using "
            "the one-standard-error rule because its validation error was within one standard error of the "
            "minimum while requiring fewer polynomial terms."
        )
    else:
        var2_selection_text = (
            f"Degree {int(raw_min2['degree'])} achieved the lowest raw cross-validation MSE. "
            f"Degree {int(selected2['degree'])} was selected using the one-standard-error rule because its "
            "validation error was within one standard error of the minimum while requiring fewer terms."
        )
    story.append(Paragraph(var2_selection_text, styles["body"]))
    story.append(Paragraph(
        f"The raw minimum was {raw_min2['mean_validation_mse']:.6f} at degree "
        f"{int(raw_min2['degree'])} (alpha {float(raw_min2['alpha']):g}, "
        f"{int(raw_min2['polynomial_terms'])} terms). The selected degree-10 error was "
        f"{selected2['mean_validation_mse']:.6f}, below the one-SE cutoff of "
        f"{selected2['one_se_cutoff_mse']:.6f}.",
        styles["small"],
    ))

    ols_rows = [["Problem", "Best OLS degree", "CV MSE", "CV R-squared"]]
    for name in PROBLEMS:
        ols = cv[name][cv[name]["family"] == "OLS"].sort_values("mean_validation_mse").iloc[0]
        ols_rows.append([
            name, str(int(ols["degree"])), f"{ols['mean_validation_mse']:.6f}",
            f"{ols['mean_validation_r2']:.6f}",
        ])
    story.append(Paragraph("OLS comparison and generalization", styles["h2"]))
    story.append(make_table(ols_rows, [90, 130, 150, 145], styles, padding=4))
    v1 = selections["var1"]["selected"]
    v1_d10 = best_ridge(cv["var1"], 10)
    v2_d11 = best_ridge(cv["var2"], 11)
    v2_d15 = best_ridge(cv["var2"], 15)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Ridge improved on the best OLS candidate for both tasks. For var1, degree 10 had lower training "
        f"MSE ({v1_d10['mean_training_mse']:.3f} vs {v1['mean_training_mse']:.3f} at selected degree 5) "
        f"but higher validation MSE ({v1_d10['mean_validation_mse']:.3f} vs {v1['mean_validation_mse']:.3f}). "
        f"For var2, from degree 11 to 15 training MSE fell ({v2_d11['mean_training_mse']:.3f} to "
        f"{v2_d15['mean_training_mse']:.3f}) while validation MSE rose "
        f"({v2_d11['mean_validation_mse']:.3f} to {v2_d15['mean_validation_mse']:.3f}).",
        styles["small"],
    ))

    story.append(Paragraph("Focused Ridge alpha refinement", styles["h2"]))
    story.append(Paragraph(
        "The local grids were var1: 1, 3, 5, 7, 10, 15, 20, 30, 50 and var2: "
        "0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5. They used the same folds and fold-local scaling. "
        "A candidate required at least 1% lower mean CV MSE and paired mean fold improvement greater "
        "than its paired standard error to be adopted.",
        styles["small"],
    ))
    refinement_rows = [["Problem", "Baseline / best alpha", "Best local MSE", "Relative gain", "Paired gain (SE)", "Decision"]]
    for name in PROBLEMS:
        refinement = selections[name]["alpha_refinement"]
        refinement_rows.append([
            name,
            f"{refinement['baseline_alpha']:g} / {refinement['best_local_alpha']:g}",
            f"{refinement['best_local_mean_validation_mse']:.6f}",
            f"{100 * refinement['relative_mse_improvement']:.2f}%",
            f"{refinement['paired_mean_mse_improvement']:.6g} ({refinement['paired_se_mse_improvement']:.3g})",
            "Accepted" if refinement["meaningful_improvement"] else f"Kept {refinement['final_alpha']:g}",
        ])
    story.append(make_table(refinement_rows, [48, 100, 91, 75, 125, 84], styles, padding=4, font_size=6.8))
    if selections["var1"]["alpha_refinement"]["full_search_rerun"]:
        story.append(Paragraph(
            "Var1 alpha 20 passed both safeguards, so its full degree search was rerun with the focused "
            "values added. Var2 retained alpha 1 because its local improvement was not meaningful.",
            styles["small"],
        ))

    story.append(Paragraph("Prediction and reproduction", styles["h2"]))
    story.append(Paragraph(
        "Each final model was fit on its full training set and predicted the matching test rows in their "
        "original order. Both output files were checked for exactly one y column, 1,000 finite predictions, "
        "and no index column: outputs/BT2024146_pred_var1.csv and outputs/BT2024146_pred_var2.csv. "
        "No hidden-test score is reported. Place the five instructor CSVs in data/ "
        "as listed in README.md, then run <font face='Courier'>python src/train_and_predict.py</font> and "
        "<font face='Courier'>python src/build_report.py</font>. Code and artifacts: "
        "<link href='https://github.com/Varun576253/ML_Assignment' color='#2F6B91'>GitHub repository</link>.",
        styles["small"],
    ))

    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return OUT_PDF


if __name__ == "__main__":
    print(build())
