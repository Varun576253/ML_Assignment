"""Build the concise assignment report from the current model outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
REPORT_PATH = REPORT_DIR / "report.pdf"

INK = colors.HexColor("#172B4D")
BLUE = colors.HexColor("#215A8E")
PALE_BLUE = colors.HexColor("#EAF1F8")
MID_GREY = colors.HexColor("#536579")
LIGHT_GREY = colors.HexColor("#D9E1EA")


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prediction(path: Path) -> int:
    """Check the submission schema and all prediction values before reporting."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header != ["y"]:
            raise ValueError(f"{path.name} must contain exactly one column named 'y'.")
        count = 0
        for row in reader:
            if len(row) != 1:
                raise ValueError(f"{path.name} contains an unexpected index or extra column.")
            value = float(row[0])
            if not math.isfinite(value):
                raise ValueError(f"{path.name} contains a non-finite prediction.")
            count += 1
    if count != 1000:
        raise ValueError(f"{path.name} must contain exactly 1000 predictions; found {count}.")
    return count


def get_cv_row(variable: str, degree: int, family: str, alpha: float) -> pd.Series:
    path = OUTPUT_DIR / f"model_selection_{variable}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required model-selection file not found: {path}")
    results = pd.read_csv(path)
    matches = results[
        (results["family"] == family)
        & (results["degree"].astype(int) == degree)
        & (pd.to_numeric(results["alpha"], errors="coerce").sub(alpha).abs() < 1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {family} degree {degree}, alpha {alpha:g} row in {path.name}."
        )
    return matches.iloc[0]


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle(
            "Eyebrow", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=BLUE, spaceAfter=5,
        ),
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=26, alignment=TA_LEFT, textColor=INK,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontSize=10, leading=14,
            textColor=MID_GREY, spaceAfter=12,
        ),
        "section": ParagraphStyle(
            "Section", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=BLUE, spaceBefore=7, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.1, leading=13, textColor=INK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.2, leading=11, textColor=MID_GREY, spaceAfter=4,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=8.2, leading=10, textColor=colors.white,
        ),
        "table": ParagraphStyle(
            "TableBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.4, leading=11, textColor=INK,
        ),
        "table_small": ParagraphStyle(
            "TableSmall", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.8, leading=10, textColor=INK,
        ),
        "center": ParagraphStyle(
            "Center", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.2, leading=10, alignment=TA_CENTER, textColor=INK,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def styled_table(
    rows: list[list[Any]], widths: list[float], *, repeat_rows: int = 1
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, BLUE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, LIGHT_GREY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
            ]
        )
    )
    return table


def page_chrome(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LIGHT_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(document.leftMargin, 17 * mm, width - document.rightMargin, 17 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(document.leftMargin, 11.5 * mm, "BT2024146  |  ML Assignment 1")
    canvas.drawRightString(width - document.rightMargin, 11.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    summary = load_json(OUTPUT_DIR / "selection_summary.json")
    diagnostics = load_json(OUTPUT_DIR / "data_diagnostics.json")
    by_variable = {record["variable"]: record for record in diagnostics}
    problems = summary["problems"]
    validation = summary["validation"]
    for variable in ("var1", "var2"):
        diagnostic = by_variable[variable]
        if not diagnostic["train_schema_matches_test_features"]:
            raise ValueError(f"Input feature schemas do not match for {variable}.")
        if any(diagnostic["train_missing_by_column"].values()) or any(
            diagnostic["test_missing_by_column"].values()
        ):
            raise ValueError(f"Missing input values were recorded for {variable}.")
    feature_bounds = [
        bound
        for diagnostic in diagnostics
        for split_name in ("feature_ranges_train", "feature_ranges_test")
        for bound in diagnostic[split_name].values()
    ]
    feature_min = min(float(bound[0]) for bound in feature_bounds)
    feature_max = max(float(bound[1]) for bound in feature_bounds)

    selected: dict[str, dict[str, Any]] = {}
    for variable, expected_degree, expected_alpha in (("var1", 5, 20.0), ("var2", 10, 1.0)):
        record = problems[variable]["selected"]
        if (
            record["family"] != "Ridge"
            or int(record["degree"]) != expected_degree
            or not math.isclose(float(record["alpha"]), expected_alpha)
        ):
            raise ValueError(f"Current selected model for {variable} does not match the final solution.")
        selected[variable] = record

    var2_raw = pd.read_csv(OUTPUT_DIR / "model_selection_var2.csv")
    raw_min = var2_raw.loc[var2_raw["mean_validation_mse"].idxmin()]
    var2_degree10 = get_cv_row("var2", 10, "Ridge", 1.0)
    var2_degree11 = get_cv_row("var2", 11, "Ridge", 1.0)
    if int(raw_min["degree"]) != 11:
        raise ValueError("The current var2 raw minimum is not degree 11; review the result files.")
    if int(var2_degree10["polynomial_terms"]) >= int(var2_degree11["polynomial_terms"]):
        raise ValueError("Expected degree 10 to use fewer polynomial terms than degree 11.")

    prediction_counts = {
        variable: validate_prediction(OUTPUT_DIR / problems[variable]["prediction_file"])
        for variable in ("var1", "var2")
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="ML Assignment 1 — Polynomial Regression",
        author="BT2024146",
        subject="Polynomial regression methodology and final validation results",
    )
    s = styles()
    story: list[Any] = []

    # Page 1: task, methodology, and validated model choices.
    fold_count = int(validation["n_splits"]) * int(validation["n_repeats"])
    train_rows = int(round(1000 * float(validation["train_fraction_per_fold"])))
    validation_rows_count = int(round(1000 * float(validation["validation_fraction_per_fold"])))
    story.extend(
        [
            para("MACHINE LEARNING ASSIGNMENT 1", s["eyebrow"]),
            para("Polynomial Regression", s["title"]),
            para("Final methodology and validation results  |  Student BT2024146", s["subtitle"]),
            para("1. Problem and data", s["section"]),
            para(
                "The assignment contains two separate supervised regression problems. Each model predicts the continuous target <b>y</b> from its assigned input features. "
                "Both datasets contain 1,000 labeled training rows and 1,000 test rows. "
                "<b>var1</b> has six features and permits polynomial degrees 1 to 10; "
                "<b>var2</b> has three features and permits degrees 1 to 20. Test labels are unavailable, so no test-set error is reported.",
                s["body"],
            ),
            para("2. Polynomial-regression methodology", s["section"]),
            para(
                "PolynomialFeatures expands the inputs to include monomials up to the candidate total degree. Ridge regression is evaluated across the full permitted degree range, with its penalty strength selected from the candidate values in the model-selection code. "
                "Ordinary least-squares polynomial regression is also compared when the expanded design contains fewer than 800 terms. "
                "The scaler is fitted separately on each fold's training partition and then applied to that fold's validation partition, preventing preprocessing leakage.",
                s["body"],
            ),
            para("3. Validation and selection", s["section"]),
        ]
    )
    validation_rows = [
        [para("Validation setting", s["table_head"]), para("Configuration", s["table_head"])],
        [para("Strategy", s["table"]), para(f"Repeated {validation['n_splits']}-fold cross-validation; {validation['n_repeats']} repeats ({fold_count} folds)", s["table"])],
        [para("Randomization", s["table"]), para(f"{'Shuffled' if validation['shuffle'] else 'Unshuffled'} folds; random seed {validation['random_seed']}", s["table"])],
        [para("Rows per fold", s["table"]), para(f"{train_rows} training rows and {validation_rows_count} validation rows", s["table"])],
        [para("Selection", s["table"]), para("Lowest degree within one standard error of the minimum mean validation MSE", s["table"])],
    ]
    story.extend(
        [
            styled_table(validation_rows, [39 * mm, 130 * mm]),
            Spacer(1, 5),
            para("Final selected models", s["section"]),
        ]
    )
    result_rows: list[list[Any]] = [
        [
            para("Problem", s["table_head"]),
            para("Model", s["table_head"]),
            para("Degree", s["table_head"]),
            para("Alpha", s["table_head"]),
            para("CV MSE", s["table_head"]),
            para("CV R-squared", s["table_head"]),
        ]
    ]
    for variable in ("var1", "var2"):
        result = selected[variable]
        result_rows.append(
            [
                para(variable, s["table"]),
                para("Ridge polynomial", s["table"]),
                para(str(result["degree"]), s["center"]),
                para(f"{float(result['alpha']):g}", s["center"]),
                para(fmt(result["mean_validation_mse"]), s["center"]),
                para(fmt(result["mean_validation_r2"]), s["center"]),
            ]
        )
    story.extend([styled_table(result_rows, [20 * mm, 42 * mm, 19 * mm, 17 * mm, 30 * mm, 36 * mm]), Spacer(1, 5)])
    story.append(
        para(
            "Degree 11 achieved the lowest raw cross-validation MSE. However, degree 10 was selected using the one-standard-error rule because its validation error was within one standard error of the minimum while requiring fewer polynomial terms.",
            s["body"],
        )
    )
    story.append(
        para(
            f"For var2, the raw minimum was {fmt(raw_min['mean_validation_mse'])} at degree 11; "
            f"the selected degree 10 result was {fmt(selected['var2']['mean_validation_mse'])}, below the one-standard-error cutoff of {fmt(selected['var2']['one_se_cutoff_mse'])}. "
            f"Degree 10 uses {int(var2_degree10['polynomial_terms'])} polynomial terms, compared with {int(var2_degree11['polynomial_terms'])} at degree 11.",
            s["small"],
        )
    )
    story.append(
        para(
            "The training-fold errors are lower than the validation errors for both selected models, showing a training-to-validation gap. Selection is therefore based on held-out fold performance and a lower-complexity degree preference, rather than training fit alone.",
            s["small"],
        )
    )

    # Page 2: data checks, final inference, and reproducibility.
    story.extend([PageBreak(), para("4. Final fitting and prediction files", s["section"])])
    story.append(
        para(
            "After model selection, each selected polynomial Ridge pipeline is refitted using all 1,000 labeled rows for its problem. It then predicts the matching test dataset, preserving that file's row order. The two problems are trained and predicted independently.",
            s["body"],
        )
    )
    submission_rows = [
        [para("File", s["table_head"]), para("Verification", s["table_head"])],
        [
            para("BT2024146_pred_var1.csv", s["table_small"]),
            para(f"{prediction_counts['var1']:,} finite predictions; one column named y; no index column", s["table_small"]),
        ],
        [
            para("BT2024146_pred_var2.csv", s["table_small"]),
            para(f"{prediction_counts['var2']:,} finite predictions; one column named y; no index column", s["table_small"]),
        ],
    ]
    story.extend([styled_table(submission_rows, [61 * mm, 103 * mm]), Spacer(1, 6)])
    story.extend(
        [
            para("5. Input checks", s["section"]),
            para(
                f"The training and test feature schemas match for both problems. The supplied data contain no missing values. Across the supplied feature columns, values range from {fmt(feature_min)} to {fmt(feature_max)}. The prediction files were checked for their required single-column schema, 1,000-row count, and finite numeric values.",
                s["body"],
            ),
            para("6. Reproducibility", s["section"]),
            para(
                "Place the five instructor-provided CSV files in the project's <b>data/</b> directory: the train and test files for var1 and var2, plus sample_submission.csv. These inputs and the assignment PDF are not required repository deliverables. Install the pinned packages from requirements.txt, then run the following from the repository root:",
                s["body"],
            ),
            para(
                "python src/train_and_predict.py<br/>python src/build_report.py",
                ParagraphStyle(
                    "Commands", parent=s["body"], fontName="Courier",
                    fontSize=8.5, leading=13, leftIndent=8, textColor=INK,
                    backColor=PALE_BLUE, borderColor=LIGHT_GREY, borderWidth=0.5,
                    borderPadding=7, spaceAfter=8,
                ),
            ),
            para(
                "The first command runs degree and model selection, refits the selected models, writes the prediction CSVs, and saves validation and data-check results under outputs/. The second builds this report from those result files. See README.md for the complete Windows PowerShell environment setup and package installation commands.",
                s["body"],
            ),
            para(
                "Code locations: src/model_selection.py contains the fold-safe model search; src/train_and_predict.py performs data checks, final training, inference, and prediction-file validation; src/build_report.py generates this PDF from the saved results.",
                s["small"],
            ),
        ]
    )

    document.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return REPORT_PATH


if __name__ == "__main__":
    print(f"Created {build_report()}")
