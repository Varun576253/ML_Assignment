# Machine Learning Assignment 1: Polynomial Regression

**Student roll number:** BT2024146

This project follows the supplied `data/ML_assignment.pdf` and uses the assigned train/test CSVs copied unchanged into `data/`. The two problems are modeled independently with polynomial regression. No test labels are used.

## Assignment scope

| Problem | Features | Target | Maximum degree |
|---|---:|---|---:|
| var1 | 6 (`x1` to `x6`) | `y` - Net Power Score | 10 |
| var2 | 3 (`x1` to `x3`) | `y` - Thermal Anomaly Score | 20 |

## Selected models

Model selection used shuffled 5-fold cross-validation repeated twice (10 validation folds total, seed 2026). The reported MSE and R-squared are mean validation-fold scores. The one-standard-error rule selects the lowest degree within one fold-to-fold standard error of the minimum mean validation MSE.

| Problem | Model | Degree | Ridge alpha | CV MSE | CV R-squared |
|---|---|---:|---:|---:|---:|
| var1 | Ridge polynomial regression | 5 | 10 | 0.53845 | 0.94930 |
| var2 | Ridge polynomial regression | 10 | 1 | 0.26225 | 0.99419 |

Every allowed degree was evaluated. Ridge alpha values from 0.01 to 100000 were compared for each degree. Unregularized polynomial least squares was also evaluated wherever the expanded feature count was below the 800 observations in a training fold. Polynomial expansion and `StandardScaler` are placed in a scikit-learn pipeline for final fitting; each validation fold fits its scaler using only that fold's training rows.

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

- `data/`: assigned datasets, sample submission, and a copy of the official assignment PDF.
- `src/model_selection.py`: polynomial model search, repeated-fold validation, and model selection.
- `src/train_and_predict.py`: data checks, final training, prediction generation, and submission validation.
- `src/build_report.py`: creates the concise PDF report from the measured run outputs.
- `outputs/`: prediction CSVs, fold-level validation summaries, diagnostics, and selected configuration.
- `report/`: final report PDF.

The prediction files contain exactly one `y` column, 1000 rows, no index column, and predictions in the original test-row order. Test labels are unavailable; no hidden-test performance is claimed.
