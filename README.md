# Machine Learning Assignment 1: Polynomial Regression

**Student roll number:** BT2024146

This project implements the polynomial-regression requirements for assignment 1. The two problems are modeled independently, and no test labels are used. Instructor-provided datasets and the assignment PDF are kept out of the GitHub deliverables.

## Assignment scope

| Problem | Features | Target | Maximum degree |
|---|---:|---|---:|
| var1 | 6 (`x1` to `x6`) | `y` - Net Power Score | 10 |
| var2 | 3 (`x1` to `x3`) | `y` - Thermal Anomaly Score | 20 |

## Selected models

Model selection uses shuffled 5-fold cross-validation repeated twice (10 validation folds total, seed 2026). The reported MSE and R-squared are mean validation-fold scores. The one-standard-error rule selects the lowest degree within one fold-to-fold standard error of the minimum mean validation MSE. The full permitted degree ranges are searched. Ridge alpha values 0.01, 0.1, 1, 10, 100, 1,000, 10,000, and 100,000 are compared across degrees, followed by a focused local alpha refinement at the selected degree using the same folds. A refinement is adopted only if it improves mean validation MSE by at least 1% and the paired mean fold improvement exceeds its paired standard error; when accepted, the complete degree search is rerun with the refined values included.

| Problem | Model | Degree | Ridge alpha | CV MSE | CV R-squared |
|---|---|---:|---:|---:|---:|
| var1 | Ridge polynomial regression | 5 | 20 | 0.51516 | 0.95153 |
| var2 | Ridge polynomial regression | 10 | 1 | 0.26225 | 0.99419 |

Unregularized polynomial least squares was also evaluated wherever the expanded feature count was below the 800 observations in a training fold. Polynomial expansion and `StandardScaler` are placed in a scikit-learn pipeline for final fitting; each validation fold fits its scaler using only that fold's training rows. For var2, degree 11 has the lowest raw cross-validation MSE; degree 10 is selected by the one-standard-error rule because it is within one standard error of the minimum and uses fewer terms.

## Input data

Before running the code, place the instructor-provided files in `data/` with these exact names:

```text
data/BT2024146_train_var1.csv
data/BT2024146_test_var1.csv
data/BT2024146_train_var2.csv
data/BT2024146_test_var2.csv
data/sample_submission.csv
```

The assignment PDF is not needed to run the code. These input files are not committed to this repository; the local `data/` copies, when present, are ignored by Git. The scripts do not modify input files.

## Reproduce

Use Python 3.12, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train_and_predict.py
python src/build_report.py
```

The training script writes both submission files, the full degree/model comparison tables, data diagnostics, and the model-selection summary under `outputs/`. The report is written to `report/report.pdf`.

## Project contents

- `data/` (local input only): place the instructor datasets here before running; its CSV and PDF files are ignored by Git.
- `src/model_selection.py`: polynomial model search, repeated-fold validation, and model selection.
- `src/train_and_predict.py`: data checks, final training, prediction generation, and submission validation.
- `src/build_report.py`: creates the concise PDF report from the measured run outputs.
- `outputs/`: prediction CSVs, fold-level validation summaries, focused alpha-refinement results, diagnostics, and selected configuration.
- `report/`: final report PDF.

The prediction files contain exactly one `y` column, 1000 rows, no index column, and predictions in the original test-row order. Test labels are unavailable; no hidden-test performance is claimed.
