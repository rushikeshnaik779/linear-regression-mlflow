"""The model variants compared in each training run. Each variant is one MLflow run."""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, RANDOM_STATE
from src.features import build_preprocessor
from src.tracking import log_run


def run_all(X_train, X_test, y_train, y_test, data_hash) -> list[dict]:
    """Run every variant and return a summary dict per run."""
    common = (X_train, y_train, X_test, y_test, data_hash)
    results = []

    # 1. Baseline: plain linear regression
    results.append(log_run(
        "baseline_linear",
        Pipeline([("prep", build_preprocessor(1)), ("model", LinearRegression())]),
        {"model": "LinearRegression", "degree": 1},
        *common,
    ))

    # 2. Add degree-2 polynomial and interaction features
    results.append(log_run(
        "poly2_linear",
        Pipeline([("prep", build_preprocessor(2)), ("model", LinearRegression())]),
        {"model": "LinearRegression", "degree": 2},
        *common,
    ))

    # 3. Same features, with Ridge regularisation
    results.append(log_run(
        "poly2_ridge",
        Pipeline([("prep", build_preprocessor(2)), ("model", Ridge(alpha=1.0))]),
        {"model": "Ridge", "degree": 2, "alpha": 1.0},
        *common,
    ))

    # 4. Feature engineering: log(distance) captures the diminishing effect of distance
    results.append(log_run(
        "logdist_linear",
        Pipeline([("prep", build_preprocessor(1, log_distance=True)), ("model", LinearRegression())]),
        {"model": "LinearRegression", "degree": 1, "log_distance": True},
        *common,
    ))

    # 5. GridSearchCV chooses polynomial degree and Ridge alpha, with log(distance)
    grid = GridSearchCV(
        Pipeline([("prep", build_preprocessor(1, log_distance=True)), ("model", Ridge())]),
        param_grid={
            "prep__num__poly__degree": [1, 2, 3],
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    best = grid.best_params_
    results.append(log_run(
        "tuned_ridge_grid",
        grid.best_estimator_,
        {
            "model": "Ridge",
            "degree": best["prep__num__poly__degree"],
            "alpha": best["model__alpha"],
            "log_distance": True,
            "search": f"GridSearchCV({CV_FOLDS}-fold)",
        },
        *common,
    ))

    return results
