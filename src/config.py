"""Project-wide constants. Paths are resolved from the project root, not the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "houses.csv"
EXPORT_DIR = PROJECT_ROOT / "models" / "house_price_champion"

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "house-price-linear-regression"

REGISTERED_MODEL_NAME = "house-price-model"
CHAMPION_ALIAS = "champion"

NUM_FEATURES = ["sqft", "bedrooms", "age_years", "distance_km"]
CAT_FEATURES = ["neighborhood"]
TARGET = "price"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
