"""MLflow logging: per-run metrics and artifacts, plus champion registration and export."""

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate

from src.config import (
    CHAMPION_ALIAS,
    CV_FOLDS,
    EXPORT_DIR,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
)


def evaluate(estimator, X_train, y_train, X_test, y_test) -> dict:
    """Cross-validate on the training set, fit on it, then score the held-out test set."""
    cv_scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=["neg_root_mean_squared_error", "r2"],
    )
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)

    rmse_scores = -cv_scores["test_neg_root_mean_squared_error"]
    return {
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
        "cv_r2_mean": cv_scores["test_r2"].mean(),
        "test_rmse": root_mean_squared_error(y_test, preds),
        "test_mae": mean_absolute_error(y_test, preds),
        "test_r2": r2_score(y_test, preds),
    }


def log_run(run_name, estimator, params, X_train, y_train, X_test, y_test, data_hash) -> dict:
    """Train and evaluate one variant, log it to MLflow, and return a summary dict."""
    metrics = evaluate(estimator, X_train, y_train, X_test, y_test)

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
        run_id = run.info.run_id

    print(
        f"{run_name:<18} test_rmse={metrics['test_rmse']:>10,.0f}  "
        f"test_r2={metrics['test_r2']:.4f}  cv_rmse={metrics['cv_rmse_mean']:>10,.0f}"
    )
    return {"name": run_name, "run_id": run_id, "test_rmse": metrics["test_rmse"]}


def promote_champion(best: dict) -> None:
    """Register the best run, give it the champion alias, and export it to models/."""
    model_uri = f"runs:/{best['run_id']}/model"

    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    mlflow.MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, mv.version
    )

    EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(EXPORT_DIR))

    print(
        f"\nChampion: run '{best['name']}' (test_rmse={best['test_rmse']:,.0f})"
        f"\n  registered as {REGISTERED_MODEL_NAME} v{mv.version} with alias '{CHAMPION_ALIAS}'"
        f"\n  exported to {EXPORT_DIR}/"
        f"\n\nServe it with:\n  mlflow models serve -m models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS} "
        f"-p 5001 --no-conda"
    )
