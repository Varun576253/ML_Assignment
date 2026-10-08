"""Select polynomial models by CV, refit on all training rows, and predict."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from model_selection import RANDOM_SEED, make_pipeline, select_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROBLEMS = {"var1": 10, "var2": 20}


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def diagnose(train: pd.DataFrame, test: pd.DataFrame, variable: str) -> dict[str, Any]:
    feature_columns = [column for column in train.columns if column != "y"]
    return {
        "variable": variable,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_columns": train.columns.tolist(),
        "test_columns": test.columns.tolist(),
        "feature_columns": feature_columns,
        "train_schema_matches_test_features": feature_columns == test.columns.tolist(),
        "train_missing_by_column": train.isna().sum().to_dict(),
        "test_missing_by_column": test.isna().sum().to_dict(),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "test_duplicate_feature_rows": int(test.duplicated().sum()),
        "feature_ranges_train": {
            column: [float(train[column].min()), float(train[column].max())]
            for column in feature_columns
        },
        "feature_ranges_test": {
            column: [float(test[column].min()), float(test[column].max())]
            for column in feature_columns
        },
        "target_summary": {
            "min": float(train["y"].min()),
            "q01": float(train["y"].quantile(0.01)),
            "q25": float(train["y"].quantile(0.25)),
            "median": float(train["y"].median()),
            "mean": float(train["y"].mean()),
            "std": float(train["y"].std(ddof=1)),
            "q75": float(train["y"].quantile(0.75)),
            "q99": float(train["y"].quantile(0.99)),
            "max": float(train["y"].max()),
        },
    }


def _check_inputs(train: pd.DataFrame, test: pd.DataFrame, variable: str) -> list[str]:
    if "y" not in train.columns:
        raise ValueError(f"Training file for {variable} is missing target column 'y'.")
    feature_columns = [column for column in train.columns if column != "y"]
    if test.columns.tolist() != feature_columns:
        raise ValueError(
            f"Train/test feature schema mismatch for {variable}: "
            f"{feature_columns} vs {test.columns.tolist()}"
        )
    if train.empty or test.empty:
        raise ValueError(f"Empty data supplied for {variable}.")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError(f"Missing values supplied for {variable}.")
    if not np.isfinite(train.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite training values supplied for {variable}.")
    if not np.isfinite(test.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite test values supplied for {variable}.")
    return feature_columns


def run(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if sample.columns.tolist() != ["y"] or len(sample) != 1000:
        raise ValueError("sample_submission.csv must contain 1000 rows in one 'y' column.")

    report: dict[str, Any] = {
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "validation": {
            "strategy": "RepeatedKFold",
            "n_splits": 5,
            "n_repeats": 2,
            "n_validation_folds": 10,
            "train_fraction_per_fold": 0.8,
            "validation_fraction_per_fold": 0.2,
            "shuffle": True,
            "random_seed": RANDOM_SEED,
            "selection_rule": "lowest degree within one fold-to-fold SE of minimum mean validation MSE",
        },
        "sample_submission_rows": int(len(sample)),
        "problems": {},
    }
    diagnostic_records: list[dict[str, Any]] = []

    for variable, max_degree in PROBLEMS.items():
        train_path = data_dir / f"BT2024146_train_{variable}.csv"
        test_path = data_dir / f"BT2024146_test_{variable}.csv"
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        feature_columns = _check_inputs(train, test, variable)
        if len(test) != len(sample):
            raise ValueError(
                f"{variable} test rows ({len(test)}) do not match the sample submission ({len(sample)})."
            )

        diagnostics = diagnose(train, test, variable)
        if not diagnostics["train_schema_matches_test_features"]:
            raise ValueError(f"Train/test schema mismatch for {variable}.")
        diagnostic_records.append(diagnostics)

        X_train = train[feature_columns]
        y_train = train["y"]
        X_test = test[feature_columns]
        selected, cv_results = select_model(
            X_train,
            y_train,
            max_degree=max_degree,
            label=variable,
        )
        cv_path = output_dir / f"model_selection_{variable}.csv"
        cv_results.to_csv(cv_path, index=False, float_format="%.12g")

        final_model = make_pipeline(selected["family"], selected["degree"], selected["alpha"])
        final_model.fit(X_train, y_train)
        predictions = np.asarray(final_model.predict(X_test), dtype=float)
        if predictions.shape != (len(test),):
            raise ValueError(f"Unexpected prediction shape for {variable}: {predictions.shape}")
        if not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite predictions generated for {variable}.")

        filename = f"BT2024146_pred_{variable}.csv"
        prediction_path = output_dir / filename
        pd.DataFrame({"y": predictions}).to_csv(
            prediction_path,
            index=False,
            float_format="%.12f",
        )
        written = pd.read_csv(prediction_path)
        if written.columns.tolist() != ["y"] or len(written) != 1000:
            raise ValueError(f"Invalid prediction CSV schema or row count: {prediction_path}")
        if not np.isfinite(written["y"].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid prediction values written to {prediction_path}")

        report["problems"][variable] = {
            "max_allowed_degree": max_degree,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(len(feature_columns)),
            "features": feature_columns,
            "selected": selected,
            "cv_results_file": cv_path.name,
            "prediction_file": filename,
            "prediction_min": float(predictions.min()),
            "prediction_max": float(predictions.max()),
            "prediction_mean": float(predictions.mean()),
            "prediction_std": float(predictions.std(ddof=1)),
            "prediction_rows": int(len(predictions)),
            "prediction_column": "y",
        }
        print(
            f"{variable}: {selected['family']} degree={selected['degree']} "
            f"alpha={selected['alpha']} MSE={selected['mean_validation_mse']:.8g} "
            f"R2={selected['mean_validation_r2']:.8g} -> {prediction_path}"
        )

    (output_dir / "data_diagnostics.json").write_text(
        json.dumps(diagnostic_records, indent=2, default=_json_value), encoding="utf-8"
    )
    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, default=_json_value), encoding="utf-8")
    print(f"Saved model selection summary to {summary_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
