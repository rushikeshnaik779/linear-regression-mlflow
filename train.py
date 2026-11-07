"""Train and compare linear regression variants, logging every run to MLflow.

Variants (each is one MLflow run):
  1. baseline_linear      - plain LinearRegression on the raw features
  2. poly2_linear         - adds degree-2 polynomial/interaction features
  3. poly2_ridge          - same features, Ridge regularisation (alpha=1.0)
  4. tuned_ridge_grid     - GridSearchCV over polynomial degree and Ridge alpha

View results with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

DATA_PATH = Path("data/houses.csv")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "house-price-linear-regression"

NUM_FEATURES = ["sqft", "bedrooms", "age_years", "distance_km"]
CAT_FEATURES = ["neighborhood"]
TARGET = "price"
RANDOM_STATE = 42

REGISTERED_MODEL_NAME = "house-price-model"
CHAMPION_ALIAS = "champion"
EXPORT_DIR = Path("models/house_price_champion")

RUNS: list[dict] = []  # filled by log_run, used to pick the champion


def build_preprocessor(degree: int = 1, log_distance: bool = False) -> ColumnTransformer:
    """Impute -> polynomial expansion -> scale for numeric; one-hot for categorical.

    degree=1 makes PolynomialFeatures an identity transform, so the same
    pipeline shape works for every variant.
    log_distance=True adds a log(1 + distance) feature, a hand-crafted
    transform that matches the nonlinear distance effect in the data.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
        ]
    )
    transformers = [
        ("num", numeric, NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ]
    if log_distance:
        transformers.append(
            (
                "log_dist",
                Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p)),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["distance_km"],
            )
        )
    return ColumnTransformer(transformers)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_run(run_name, estimator, params, X_train, y_train, X_test, y_test, data_hash):
    """Cross-validate on the training set, fit, evaluate on the test set, log all of it."""
    cv_scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring=["neg_root_mean_squared_error", "r2"],
    )

    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)

    metrics = {
        "cv_rmse_mean": -cv_scores["test_neg_root_mean_squared_error"].mean(),
        "cv_rmse_std": cv_scores["test_neg_root_mean_squared_error"].std(),
        "cv_r2_mean": cv_scores["test_r2"].mean(),
        "test_rmse": root_mean_squared_error(y_test, preds),
        "test_mae": mean_absolute_error(y_test, preds),
        "test_r2": r2_score(y_test, preds),
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("dataset_sha256", data_hash)
        mlflow.set_tag("train_rows", len(X_train))
        mlflow.set_tag("test_rows", len(X_test))
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            estimator,
            name="model",
            input_example=X_test.head(3),
            # skops flags numpy.dtype (produced by our own preprocessing) as untrusted.
            skops_trusted_types=["numpy.dtype"],
        )
        print(
            f"{run_name:<18} test_rmse={metrics['test_rmse']:>10,.0f}  "
            f"test_r2={metrics['test_r2']:.4f}  cv_rmse={metrics['cv_rmse_mean']:>10,.0f}"
        )
        RUNS.append({"name": run_name, "run_id": run.info.run_id, "test_rmse": metrics["test_rmse"]})
        return run.info.run_id, metrics


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Dataset missing. Run: python make_dataset.py")

    data_hash = file_sha256(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows\n")

    # 1. Baseline: plain linear regression
    log_run(
        "baseline_linear",
        Pipeline([("prep", build_preprocessor(1)), ("model", LinearRegression())]),
        {"model": "LinearRegression", "degree": 1},
        X_train, y_train, X_test, y_test, data_hash,
    )

    # 2. Add degree-2 polynomial and interaction features
    log_run(
        "poly2_linear",
        Pipeline([("prep", build_preprocessor(2)), ("model", LinearRegression())]),
        {"model": "LinearRegression", "degree": 2},
        X_train, y_train, X_test, y_test, data_hash,
    )

    # 3. Same features, with Ridge regularisation
    log_run(
        "poly2_ridge",
        Pipeline([("prep", build_preprocessor(2)), ("model", Ridge(alpha=1.0))]),
        {"model": "Ridge", "degree": 2, "alpha": 1.0},
        X_train, y_train, X_test, y_test, data_hash,
    )

    # 4. Feature engineering: log(distance) captures the diminishing effect of distance
    log_run(
        "logdist_linear",
        Pipeline([("prep", build_preprocessor(1, log_distance=True)), ("model", LinearRegression())]),
        {"model": "LinearRegression", "degree": 1, "log_distance": True},
        X_train, y_train, X_test, y_test, data_hash,
    )

    # 5. Let GridSearchCV choose polynomial degree and Ridge alpha, with log(distance)
    grid_pipe = Pipeline(
        [("prep", build_preprocessor(1, log_distance=True)), ("model", Ridge())]
    )
    grid = GridSearchCV(
        grid_pipe,
        param_grid={
            "prep__num__poly__degree": [1, 2, 3],
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    best = grid.best_params_
    log_run(
        "tuned_ridge_grid",
        grid.best_estimator_,
        {
            "model": "Ridge",
            "degree": best["prep__num__poly__degree"],
            "alpha": best["model__alpha"],
            "log_distance": True,
            "search": "GridSearchCV(5-fold)",
        },
        X_train, y_train, X_test, y_test, data_hash,
    )

    print(
        "\nNote: tuned_ridge_grid's CV numbers are slightly optimistic because the same"
        "\nfolds chose the hyperparameters. The test_* metrics are the honest comparison."
    )

    promote_champion()

    print(f"\nDone. Open the MLflow UI with:\n  mlflow ui --backend-store-uri {TRACKING_URI}")


def promote_champion():
    """Register the run with the lowest test RMSE, give it the 'champion' alias,
    and export it to models/ so it can be inspected or copied without MLflow."""
    best = min(RUNS, key=lambda r: r["test_rmse"])
    model_uri = f"runs:/{best['run_id']}/model"

    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, mv.version)

    EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(EXPORT_DIR))

    print(
        f"\nChampion: run '{best['name']}' (test_rmse={best['test_rmse']:,.0f})"
        f"\n  registered as {REGISTERED_MODEL_NAME} v{mv.version} with alias '{CHAMPION_ALIAS}'"
        f"\n  exported to {EXPORT_DIR}/"
        f"\n\nServe it with:\n  mlflow models serve -m models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS} "
        f"-p 5001 --no-conda"
    )


if __name__ == "__main__":
    main()
