"""Generate the assignment report from the saved final model-selection outputs."""

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
PALE_GREEN = colors.HexColor("#E6F3E9")
PALE_GOLD = colors.HexColor("#FFF4D6")
MID_GREY = colors.HexColor("#536579")
LIGHT_GREY = colors.HexColor("#D9E1EA")


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def validate_prediction(path: Path) -> int:
    """Verify the submission header, row count, and finite numeric values."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        if next(reader, None) != ["y"]:
            raise ValueError(f"{path.name} must contain exactly one column named 'y'.")
        count = 0
        for row in reader:
            if len(row) != 1:
                raise ValueError(f"{path.name} contains an unexpected index or extra column.")
            if not math.isfinite(float(row[0])):
                raise ValueError(f"{path.name} contains a non-finite prediction.")
            count += 1
    if count != 1000:
        raise ValueError(f"{path.name} must contain 1000 predictions; found {count}.")
    return count


def best_ridge_by_degree(variable: str, max_degree: int) -> pd.DataFrame:
    path = OUTPUT_DIR / f"model_selection_{variable}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required model-selection file not found: {path}")
    results = pd.read_csv(path)
    ridge = results[results["family"] == "Ridge"].copy()
    best = (
        ridge.sort_values("mean_validation_mse")
        .groupby("degree", as_index=False)
        .first()
        .sort_values("degree")
        .reset_index(drop=True)
    )
    if best["degree"].astype(int).tolist() != list(range(1, max_degree + 1)):
        raise ValueError(f"The full permitted Ridge degree range is missing for {variable}.")
    return best


def best_ols(variable: str) -> pd.Series:
    path = OUTPUT_DIR / f"model_selection_{variable}.csv"
    results = pd.read_csv(path)
    ols = results[results["family"] == "OLS"]
    if ols.empty:
        raise ValueError(f"No OLS comparison rows found for {variable}.")
    return ols.loc[ols["mean_validation_mse"].idxmin()]


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "eyebrow": ParagraphStyle(
            "Eyebrow", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=BLUE, spaceAfter=4,
        ),
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=26, alignment=TA_LEFT, textColor=INK,
            spaceAfter=3,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontSize=9.5, leading=13,
            textColor=MID_GREY, spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "Section", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=6, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.8, leading=12, textColor=INK, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.8, leading=10, textColor=MID_GREY, spaceAfter=4,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=7.6, leading=9, textColor=colors.white,
        ),
        "table": ParagraphStyle(
            "TableBody", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.7, leading=9.5, textColor=INK,
        ),
        "table_tight": ParagraphStyle(
            "TableTight", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.1, leading=8.4, textColor=INK,
        ),
        "center": ParagraphStyle(
            "Center", parent=base["BodyText"], fontName="Helvetica",
            fontSize=7.7, leading=9.2, alignment=TA_CENTER, textColor=INK,
        ),
        "code": ParagraphStyle(
            "Commands", parent=base["BodyText"], fontName="Courier",
            fontSize=8, leading=12, leftIndent=7, textColor=INK,
            backColor=PALE_BLUE, borderColor=LIGHT_GREY, borderWidth=0.5,
            borderPadding=6, spaceAfter=6,
        ),
    }


def table_style(highlights: list[tuple[int, int, colors.Color]] | None = None) -> TableStyle:
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BLUE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, LIGHT_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
    ]
    for row, start_col, fill in highlights or []:
        commands.append(("BACKGROUND", (start_col, row), (start_col + 2, row), fill))
    return TableStyle(commands)


def make_table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    highlights: list[tuple[int, int, colors.Color]] | None = None,
    repeat_rows: int = 1,
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    table.setStyle(table_style(highlights))
    return table


def page_chrome(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(document.leftMargin, A4[1] - 10 * mm, "BT2024146  |  MACHINE LEARNING ASSIGNMENT 1")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(width - document.rightMargin, A4[1] - 10 * mm, "Polynomial regression | Test labels unavailable")
    canvas.setStrokeColor(LIGHT_GREY)
    canvas.setLineWidth(0.45)
    canvas.line(document.leftMargin, 16 * mm, width - document.rightMargin, 16 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(document.leftMargin, 10.5 * mm, "Polynomial Regression  |  BT2024146")
    canvas.drawRightString(width - document.rightMargin, 10.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    summary = load_json(OUTPUT_DIR / "selection_summary.json")
    diagnostics = load_json(OUTPUT_DIR / "data_diagnostics.json")
    diagnostic_by_problem = {row["variable"]: row for row in diagnostics}
    problems = summary["problems"]
    validation = summary["validation"]
    max_degrees = {variable: int(problems[variable]["max_allowed_degree"]) for variable in ("var1", "var2")}

    for variable in ("var1", "var2"):
        diagnostic = diagnostic_by_problem[variable]
        if not diagnostic["train_schema_matches_test_features"]:
            raise ValueError(f"Training and test feature schemas differ for {variable}.")
        if any(diagnostic["train_missing_by_column"].values()) or any(
            diagnostic["test_missing_by_column"].values()
        ):
            raise ValueError(f"Missing input values are recorded for {variable}.")

    selected = {variable: problems[variable]["selected"] for variable in ("var1", "var2")}
    expected = {"var1": ("Ridge", 5, 20.0), "var2": ("Ridge", 10, 1.0)}
    for variable, (family, degree, alpha) in expected.items():
        result = selected[variable]
        if (
            result["family"] != family
            or int(result["degree"]) != degree
            or not math.isclose(float(result["alpha"]), alpha)
        ):
            raise ValueError(f"Saved selected model for {variable} does not match the final solution.")

    ridge_by_degree = {
        variable: best_ridge_by_degree(variable, max_degrees[variable])
        for variable in ("var1", "var2")
    }
    raw_var2 = pd.read_csv(OUTPUT_DIR / "model_selection_var2.csv")
    raw_var2 = raw_var2.loc[raw_var2["mean_validation_mse"].idxmin()]
    if raw_var2["family"] != "Ridge" or int(raw_var2["degree"]) != 11:
        raise ValueError("The saved var2 raw CV minimum does not match the degree-11 result.")
    ols_best = {variable: best_ols(variable) for variable in ("var1", "var2")}
    predictions = {
        variable: validate_prediction(OUTPUT_DIR / problems[variable]["prediction_file"])
        for variable in ("var1", "var2")
    }

    feature_bounds = [
        bound
        for diagnostic in diagnostics
        for range_name in ("feature_ranges_train", "feature_ranges_test")
        for bound in diagnostic[range_name].values()
    ]
    global_feature_min = min(float(bound[0]) for bound in feature_bounds)
    global_feature_max = max(float(bound[1]) for bound in feature_bounds)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=17 * mm,
        bottomMargin=21 * mm,
        title="ML Assignment 1 — Polynomial Regression",
        author="BT2024146",
        subject="Current polynomial-regression methodology and validation results",
    )
    s = make_styles()
    story: list[Any] = []

    # Page 1: assignment, data profile, validation plan, and final models.
    story.extend(
        [
            paragraph("MACHINE LEARNING ASSIGNMENT 1", s["eyebrow"]),
            paragraph("Polynomial Regression", s["title"]),
            paragraph("Final solution and cross-validation results  |  Student BT2024146", s["subtitle"]),
            paragraph("1. Assignment and data", s["section"]),
            paragraph(
                "This work solves two separate supervised regression tasks. Each model predicts the continuous target <b>y</b> from the features supplied for its problem. "
                "The permitted total polynomial degrees are 1 to 10 for var1 and 1 to 20 for var2. Each problem has 1,000 labeled training rows and 1,000 test rows; test labels are unavailable.",
                s["body"],
            ),
        ]
    )
    data_rows: list[list[Any]] = [
        [
            paragraph("Problem", s["table_head"]),
            paragraph("Input features", s["table_head"]),
            paragraph("Train / test", s["table_head"]),
            paragraph("Target y: range; mean; SD", s["table_head"]),
            paragraph("Data checks", s["table_head"]),
        ]
    ]
    for variable in ("var1", "var2"):
        diagnostic = diagnostic_by_problem[variable]
        target = diagnostic["target_summary"]
        missing = sum(diagnostic["train_missing_by_column"].values()) + sum(
            diagnostic["test_missing_by_column"].values()
        )
        feature_range = diagnostic["feature_ranges_train"]
        lower = min(float(bounds[0]) for bounds in feature_range.values())
        upper = max(float(bounds[1]) for bounds in feature_range.values())
        checks = (
            f"Missing {missing}<br/>Train duplicates {diagnostic['train_duplicate_rows']}"
            f"<br/>Test feature duplicates {diagnostic['test_duplicate_feature_rows']}"
        )
        data_rows.append(
            [
                paragraph(variable, s["table"]),
                paragraph(f"{len(diagnostic['feature_columns'])}: {', '.join(diagnostic['feature_columns'])}<br/>Range {fmt(lower, 1)} to {fmt(upper, 1)}", s["table"]),
                paragraph(f"{diagnostic['train_shape'][0]:,} / {diagnostic['test_shape'][0]:,} rows", s["table"]),
                paragraph(f"{fmt(target['min'], 2)} to {fmt(target['max'], 2)}<br/>{fmt(target['mean'], 2)}; SD {fmt(target['std'], 2)}", s["table"]),
                paragraph(checks, s["table"]),
            ]
        )
    story.extend(
        [
            make_table(data_rows, [17 * mm, 44 * mm, 27 * mm, 37 * mm, 47 * mm]),
            Spacer(1, 5),
            paragraph(
                f"Training and test feature schemas match for both problems. The supplied data have no missing values; all features span {fmt(global_feature_min, 1)} to {fmt(global_feature_max, 1)}. "
                f"The sample submission contains {int(summary['sample_submission_rows']):,} rows and is validated by the training script as one column named y.",
                s["small"],
            ),
            paragraph("2. Validation and model methodology", s["section"]),
            paragraph(
                "The 1,000 labeled rows in each problem are evaluated with shuffled RepeatedKFold cross-validation: five folds repeated twice, for 10 validation folds total (random seed 2026). Each fold uses 800 rows for training and 200 for validation. This repeated shuffled split provides several estimates of generalization while retaining the full training sample for the final fit.",
                s["body"],
            ),
            paragraph(
                "PolynomialFeatures creates all monomials up to each candidate total degree. Ridge candidates are compared across the complete permitted degree range; ordinary least-squares polynomial regression is also evaluated at degrees with fewer than 800 expanded terms. The maximum expansions contain 8,007 terms for var1 at degree 10 and 1,770 terms for var2 at degree 20, so Ridge regularization is important for controlling coefficient size at higher degrees.",
                s["body"],
            ),
            paragraph(
                "The feature expansion is deterministic. StandardScaler is fitted only on each fold's training partition and then applied to that fold's validation partition. Mean validation MSE and R-squared are calculated across the 10 folds. The selected degree is the lowest degree within one standard error of the minimum mean validation MSE; within that degree, the lowest-MSE candidate is selected.",
                s["body"],
            ),
            paragraph("Final selected models", s["section"]),
        ]
    )
    final_rows: list[list[Any]] = [
        [
            paragraph("Problem", s["table_head"]),
            paragraph("Model", s["table_head"]),
            paragraph("Degree", s["table_head"]),
            paragraph("Alpha", s["table_head"]),
            paragraph("Terms", s["table_head"]),
            paragraph("CV MSE", s["table_head"]),
            paragraph("CV R-squared", s["table_head"]),
        ]
    ]
    for variable in ("var1", "var2"):
        result = selected[variable]
        final_rows.append(
            [
                paragraph(variable, s["table"]),
                paragraph("Ridge polynomial", s["table"]),
                paragraph(str(result["degree"]), s["center"]),
                paragraph(f"{float(result['alpha']):g}", s["center"]),
                paragraph(f"{int(result['polynomial_terms']):,}", s["center"]),
                paragraph(fmt(result["mean_validation_mse"]), s["center"]),
                paragraph(fmt(result["mean_validation_r2"]), s["center"]),
            ]
        )
    story.append(make_table(final_rows, [19 * mm, 37 * mm, 17 * mm, 16 * mm, 18 * mm, 29 * mm, 36 * mm]))

    # Page 2: degree-by-degree validation evidence and OLS comparison.
    story.extend(
        [
            PageBreak(),
            paragraph("3. Degree search and model comparison", s["section"]),
            paragraph(
                "The table reports the best mean validation MSE and R-squared among Ridge candidates at each degree. Every permitted degree was evaluated for both problems. A star marks the selected degree; the dagger marks the raw minimum for var2.",
                s["body"],
            ),
        ]
    )
    var1_by_degree = ridge_by_degree["var1"].set_index("degree")
    var2_by_degree = ridge_by_degree["var2"].set_index("degree")
    degree_rows: list[list[Any]] = [
        [
            paragraph("var1 degree", s["table_head"]),
            paragraph("CV MSE", s["table_head"]),
            paragraph("CV R-squared", s["table_head"]),
            paragraph("var2 degree", s["table_head"]),
            paragraph("CV MSE", s["table_head"]),
            paragraph("CV R-squared", s["table_head"]),
        ]
    ]
    degree_highlights: list[tuple[int, int, colors.Color]] = []
    for degree in range(1, max(max_degrees.values()) + 1):
        row_number = degree
        if degree <= max_degrees["var1"]:
            row1 = var1_by_degree.loc[degree]
            mark1 = " *" if degree == int(selected["var1"]["degree"]) else ""
            left = [
                paragraph(f"{degree}{mark1}", s["center"]),
                paragraph(fmt(row1["mean_validation_mse"], 4), s["center"]),
                paragraph(fmt(row1["mean_validation_r2"], 4), s["center"]),
            ]
            if mark1:
                degree_highlights.append((row_number, 0, PALE_GREEN))
        else:
            left = [paragraph("-", s["center"])] * 3
        row2 = var2_by_degree.loc[degree]
        mark2 = " *" if degree == int(selected["var2"]["degree"]) else (" dagger" if degree == int(raw_var2["degree"]) else "")
        label2 = f"{degree}*" if mark2 == " *" else (f"{degree}\u2020" if mark2 else str(degree))
        right = [
            paragraph(label2, s["center"]),
            paragraph(fmt(row2["mean_validation_mse"], 4), s["center"]),
            paragraph(fmt(row2["mean_validation_r2"], 4), s["center"]),
        ]
        if mark2:
            degree_highlights.append((row_number, 3, PALE_GREEN if "*" in label2 else PALE_GOLD))
        degree_rows.append(left + right)
    story.append(
        make_table(
            degree_rows,
            [17 * mm, 28 * mm, 27 * mm, 17 * mm, 28 * mm, 27 * mm],
            highlights=degree_highlights,
        )
    )
    story.append(
        paragraph(
            "* Selected degree. † Lowest raw mean CV MSE for var2. Scores are means over the same 10 validation folds.",
            s["small"],
        )
    )
    story.append(
        paragraph(
            "Degree 11 achieved the lowest raw cross-validation MSE. However, degree 10 was selected using the one-standard-error rule because its validation error was within one standard error of the minimum while requiring fewer polynomial terms.",
            s["body"],
        )
    )
    story.append(
        paragraph(
            f"For var2, degree 11 had mean CV MSE {fmt(raw_var2['mean_validation_mse'])}; selected degree 10 had {fmt(selected['var2']['mean_validation_mse'])}, within the one-standard-error cutoff of {fmt(selected['var2']['one_se_cutoff_mse'])}. Degree 10 uses {int(selected['var2']['polynomial_terms'])} terms versus {int(var2_by_degree.loc[11]['polynomial_terms'])} at degree 11.",
            s["small"],
        )
    )
    story.append(paragraph("OLS benchmark", s["section"]))
    ols_rows: list[list[Any]] = [
        [
            paragraph("Problem", s["table_head"]),
            paragraph("Best OLS degree", s["table_head"]),
            paragraph("Terms", s["table_head"]),
            paragraph("CV MSE", s["table_head"]),
            paragraph("CV R-squared", s["table_head"]),
            paragraph("Selected Ridge CV MSE", s["table_head"]),
        ]
    ]
    for variable in ("var1", "var2"):
        result = ols_best[variable]
        ols_rows.append(
            [
                paragraph(variable, s["table"]),
                paragraph(str(int(result["degree"])), s["center"]),
                paragraph(f"{int(result['polynomial_terms']):,}", s["center"]),
                paragraph(fmt(result["mean_validation_mse"]), s["center"]),
                paragraph(fmt(result["mean_validation_r2"]), s["center"]),
                paragraph(fmt(selected[variable]["mean_validation_mse"]), s["center"]),
            ]
        )
    story.append(make_table(ols_rows, [20 * mm, 28 * mm, 20 * mm, 32 * mm, 34 * mm, 38 * mm]))
    story.append(
        paragraph(
            "The selected Ridge pipeline has lower mean validation MSE than the best evaluated OLS polynomial for both problems. The final choices use Ridge degree 5, alpha 20 for var1 and Ridge degree 10, alpha 1 for var2.",
            s["small"],
        )
    )

    # Page 3: generalization evidence, final inference, and reproduction.
    story.extend(
        [
            PageBreak(),
            paragraph("4. Generalization and final model use", s["section"]),
            paragraph(
                "Training-fold and validation-fold scores for the selected models are shown below. The validation results are the model-selection estimates; no score is claimed for the hidden test labels.",
                s["body"],
            ),
        ]
    )
    generalization_rows: list[list[Any]] = [
        [
            paragraph("Problem", s["table_head"]),
            paragraph("Train MSE", s["table_head"]),
            paragraph("CV MSE", s["table_head"]),
            paragraph("Train R-squared", s["table_head"]),
            paragraph("CV R-squared", s["table_head"]),
        ]
    ]
    for variable in ("var1", "var2"):
        result = selected[variable]
        generalization_rows.append(
            [
                paragraph(variable, s["table"]),
                paragraph(fmt(result["mean_training_mse"]), s["center"]),
                paragraph(fmt(result["mean_validation_mse"]), s["center"]),
                paragraph(fmt(result["mean_training_r2"]), s["center"]),
                paragraph(fmt(result["mean_validation_r2"]), s["center"]),
            ]
        )
    story.append(make_table(generalization_rows, [25 * mm, 31 * mm, 31 * mm, 40 * mm, 42 * mm]))
    var1_max = ridge_by_degree["var1"].loc[lambda frame: frame["degree"] == max_degrees["var1"]].iloc[0]
    var1_selected = selected["var1"]
    var2_d11 = var2_by_degree.loc[11]
    var2_d15 = var2_by_degree.loc[15]
    story.append(
        paragraph(
            f"The degree search shows a training-to-validation gap at higher complexity. For var1, the degree-10 Ridge candidate's training MSE was {fmt(var1_max['mean_training_mse'])} while its validation MSE was {fmt(var1_max['mean_validation_mse'])}; the selected degree-5 candidate had training MSE {fmt(var1_selected['mean_training_mse'])} and validation MSE {fmt(var1_selected['mean_validation_mse'])}. For var2, from degree 11 to degree 15 training MSE fell from {fmt(var2_d11['mean_training_mse'])} to {fmt(var2_d15['mean_training_mse'])}, while validation MSE rose from {fmt(var2_d11['mean_validation_mse'])} to {fmt(var2_d15['mean_validation_mse'])}. These results support using validation performance and a lower-complexity choice instead of minimizing training error.",
            s["body"],
        )
    )
    story.extend(
        [
            paragraph("5. Final fitting and prediction files", s["section"]),
            paragraph(
                "After model selection, each selected pipeline is refitted using all 1,000 labeled training rows for its problem. The fitted pipeline predicts the corresponding 1,000 test rows in their supplied order. The var1 and var2 models are trained and applied independently.",
                s["body"],
            ),
        ]
    )
    prediction_rows = [
        [paragraph("Prediction file", s["table_head"]), paragraph("Automated checks", s["table_head"])],
        [paragraph("outputs/BT2024146_pred_var1.csv", s["table"]), paragraph(f"{predictions['var1']:,} finite numeric predictions; exactly one y column; no index column", s["table"])],
        [paragraph("outputs/BT2024146_pred_var2.csv", s["table"]), paragraph(f"{predictions['var2']:,} finite numeric predictions; exactly one y column; no index column", s["table"])],
    ]
    story.append(make_table(prediction_rows, [69 * mm, 99 * mm]))
    story.extend(
        [
            paragraph("6. Reproducibility", s["section"]),
            paragraph(
                "Place the five instructor-provided files in data/: BT2024146_train_var1.csv, BT2024146_test_var1.csv, BT2024146_train_var2.csv, BT2024146_test_var2.csv, and sample_submission.csv. These data files and the assignment PDF are not required GitHub deliverables. From the repository root, install requirements.txt and run:",
                s["body"],
            ),
            paragraph(
                "python -m pip install -r requirements.txt<br/>python src/train_and_predict.py<br/>python src/build_report.py",
                s["code"],
            ),
            paragraph(
                "The training script runs the degree and model search, refits the selected models, writes the prediction CSVs, and saves the validation results and data diagnostics in outputs/. The report builder reads those current outputs. README.md provides the full Windows PowerShell virtual-environment setup.",
                s["small"],
            ),
            paragraph(
                "Code: src/model_selection.py implements fold-safe polynomial model selection; src/train_and_predict.py handles input checks, final fitting, predictions, and CSV validation; src/build_report.py generates this report. Repository: github.com/Varun576253/ML_Assignment.",
                s["small"],
            ),
        ]
    )

    document.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return REPORT_PATH


if __name__ == "__main__":
    print(f"Created {build_report()}")
