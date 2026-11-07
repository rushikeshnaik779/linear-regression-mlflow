# Project rules for Claude

This project is machine learning only: linear regression with scikit-learn, tracked and served with MLflow. Do not add front-end work to this repo.

## Code layout

- All library code lives in the `src/` package: `config.py`, `data.py`, `features.py`, `experiments.py`, `tracking.py`, `pipeline.py`.
- `train.py` at the root is a thin entry point that calls `src.pipeline.main`. Keep it that way; put logic in `src/`.
- `make_dataset.py` generates `data/houses.csv`. It is seeded, so the output is reproducible.
- Paths come from `src/config.py` and resolve from the project root. Do not hard-code relative paths.

## MLflow

- Tracking store: `sqlite:///mlflow.db` (from `src/config.py`). Experiment: `house-price-linear-regression`.
- Every variant is one run. The best run by `test_rmse` is registered as `house-price-model` and tagged `champion`. The same model is exported to `models/house_price_champion/`.
- Serve with `mlflow models serve -m "models:/house-price-model@champion" -p 5001 --no-conda`.
- Do not delete `mlflow.db`, `mlruns/`, `mlartifacts/`, or registry versions without asking first. They hold experiment history.
- Compare models on `test_*` metrics. The `cv_*` numbers for `tuned_ridge_grid` are optimistic, because the same folds selected its hyperparameters.

## Git

- Commit messages end with this line:
  `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`
- Pull request descriptions end with this line:
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
- Commit or push only when the user asks. If on `main`, create a branch first.
- Generated and local-only files are git-ignored: `data/`, `models/`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `__pycache__/`, `.venv/`. Do not commit them.
- Confirm before changing repo visibility, deleting a repo, or rewriting history.
- The commit author is set in this repo's local git config only. Do not change global git config.

## Personal data

- The user's email (`aifornaik@gmail.com`) is for identification and commit authorship only. Do not send it to any other service, URL, or payload unless the user asks.
- The user's GitHub account is `rushikeshnaik779`. The repo is `linear-regression-mlflow`.

## Working style

- Verify changes by running them. Report failures with their output, and say which checks were not run.
- Put scratch files in the session scratchpad, not in the project.
- When a choice is not covered here, pick the simplest option that keeps the rules above, and state the choice in the final message.
