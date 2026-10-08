# Machine Learning Assignment 1: Polynomial Regression

**Student:** BT2024146

This repository contains the polynomial-regression code, final prediction files, model-selection results, and report for the two assigned problems. The test labels are unavailable and were not used.

## Tasks and selected models

| Problem | Features | Target | Degree limit | Selected model | CV MSE | CV R-squared |
|---|---:|---|---:|---|---:|---:|
| var1 | 6 (`x1`–`x6`) | Net Power Score (`y`) | 10 | Ridge, degree 5, alpha 20 | 0.515157 | 0.951535 |
| var2 | 3 (`x1`–`x3`) | Thermal Anomaly Score (`y`) | 20 | Ridge, degree 10, alpha 1 | 0.262251 | 0.994186 |

The scores are means over 10 validation folds from shuffled 5-fold cross-validation repeated twice (seed 2026; 800 training and 200 validation rows per fold). Every permitted polynomial degree was searched. `StandardScaler` is fit on each fold's training rows only. The one-standard-error rule picks the lowest degree within one standard error of the lowest mean CV MSE. Ridge uses the base alpha grid `0.01, 0.1, 1, 10, 100, 1,000, 10,000, 100,000` at each degree; OLS is also compared where the polynomial term count is below 800.

For var2, degree 11 had the lowest raw CV MSE (0.260706). Degree 10 was selected within the one-standard-error cutoff (0.268001) because it uses fewer polynomial terms. The report explains this choice and the validation results.

## Alpha refinement

After the degree search, focused Ridge alpha values were compared at each selected degree using the same validation folds. A candidate is adopted only when mean CV MSE improves by at least 1% and the paired mean fold improvement exceeds its paired standard error. An accepted refinement triggers a full degree-search rerun with the refined alphas added.

The focused grids are `var1: 1, 3, 5, 7, 10, 15, 20, 30, 50` and `var2: 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5`.

| Problem | Baseline alpha | Best local alpha | Relative CV MSE improvement | Outcome |
|---|---:|---:|---:|---|
| var1 | 10 | 20 | 4.33% | Adopted; full degree search rerun |
| var2 | 1 | 0.7 | 0.12% | Retained alpha 1 |

## Input data

The instructor-provided data and assignment PDF are not committed to GitHub. Put these five CSV files in the project `data/` directory before running the scripts:

```text
data/BT2024146_train_var1.csv
data/BT2024146_test_var1.csv
data/BT2024146_train_var2.csv
data/BT2024146_test_var2.csv
data/sample_submission.csv
```

The scripts read these files without modifying them. `--data-dir` can point to a different input directory.

## Reproduce

Use Python 3.12. From the project root, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train_and_predict.py
python src/build_report.py
```

The training command searches models, refits the selected pipelines on all training data, predicts test rows in their original order, and writes the prediction and result files. To use a different input directory, pass `--data-dir PATH`. To override the training output location, pass `--output-dir PATH`; the report builder reads the default project `outputs/` directory, so use the default when you want the PDF to describe the latest run.

## Outputs

- `outputs/BT2024146_pred_var1.csv` and `outputs/BT2024146_pred_var2.csv`: each has one `y` column and 1,000 finite predictions.
- `outputs/model_selection_var1.csv` and `outputs/model_selection_var2.csv`: degree/model CV comparisons and fold scores.
- `outputs/alpha_refinement_var1.csv` and `outputs/alpha_refinement_var2.csv`: focused local alpha comparisons.
- `outputs/selection_summary.json` and `outputs/data_diagnostics.json`: selected configurations, validation details, and input checks.
- `report/report.pdf`: concise assignment report.

## Code

- `src/model_selection.py`: leakage-safe repeated CV, polynomial model search, and final pipeline construction.
- `src/train_and_predict.py`: input checks, selection, final fitting, predictions, and output validation.
- `src/build_report.py`: report generation from the measured outputs.
- `requirements.txt`: pinned Python dependencies.
