# Linear Regression + MLflow: house prices

Synthetic house-price dataset, several linear regression variants, and every run tracked in MLflow.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python make_dataset.py   # writes data/houses.csv (2000 rows, some missing values)
python train.py          # trains 5 variants, logs each as an MLflow run
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://127.0.0.1:5000
```

## Model registry and serving

`train.py` picks the run with the lowest test RMSE, registers it in the MLflow Model Registry as `house-price-model`, and gives that version the `champion` alias. It also exports the model to `models/house_price_champion/model/`.

Serve the champion over REST:

```bash
mlflow models serve -m "models:/house-price-model@champion" -p 5001 --no-conda
```

Request a prediction:

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"sqft":1500,"bedrooms":3,"age_years":10,"distance_km":2.5,"neighborhood":"downtown"}]}'
```

The server must be able to reach the tracking store, so set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` if you run it from another shell.

## Variants

| Run | What changes | Test RMSE | Test R² |
|---|---|---|---|
| `baseline_linear` | Plain `LinearRegression` on raw features | 32,148 | 0.8396 |
| `poly2_linear` | Degree-2 polynomial and interaction features | 30,634 | 0.8544 |
| `poly2_ridge` | Same features, Ridge (alpha=1.0) | 30,608 | 0.8546 |
| `logdist_linear` | Adds `log(1 + distance)` feature | 29,686 | 0.8632 |
| `tuned_ridge_grid` | GridSearchCV over degree and alpha, with log-distance | 29,688 | 0.8632 |

Numbers come from the seeded dataset (seed 42), so they should reproduce exactly.

## What the results show

- The data contains a diminishing-returns distance effect, `-30000 * log1p(distance)`. Plain linear regression can't express it, and generic polynomial features only approximate it. Adding the transform directly (`logdist_linear`) gives the largest gain.
- The data's noise has σ = 25,000, so no model can get much below RMSE ≈ 25k. We are close to that floor.
- The grid search agreed with the hand-crafted feature: it picked degree 1, so the extra polynomial terms weren't useful.
- `tuned_ridge_grid`'s CV numbers are slightly optimistic, because the same folds chose its hyperparameters. Compare runs on the `test_*` metrics.

## Files

- `make_dataset.py`: generates the synthetic data.
- `train.py`: preprocessing pipelines, training, CV, and MLflow logging.
- `mlflow.db`, `mlartifacts/`: created on first run. The tracking store and model artifacts.
