"""Loading the dataset, hashing it for reproducibility, and the train/test split."""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CAT_FEATURES, DATA_PATH, NUM_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(path: Path = DATA_PATH):
    """Return (X_train, X_test, y_train, y_test, data_hash)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing at {path}. Run: python make_dataset.py")

    df = pd.read_csv(path)
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, file_sha256(path)
