"""End-to-end training: load data, run every variant, register the champion."""

import mlflow

from src.config import EXPERIMENT_NAME, TRACKING_URI
from src.data import load_split
from src.experiments import run_all
from src.tracking import promote_champion


def main() -> None:
    X_train, X_test, y_train, y_test, data_hash = load_split()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows\n")
    results = run_all(X_train, X_test, y_train, y_test, data_hash)

    print(
        "\nNote: tuned_ridge_grid's CV numbers are slightly optimistic because the same"
        "\nfolds chose the hyperparameters. The test_* metrics are the honest comparison."
    )

    promote_champion(min(results, key=lambda r: r["test_rmse"]))

    print(f"\nDone. Open the MLflow UI with:\n  mlflow ui --backend-store-uri {TRACKING_URI}")
