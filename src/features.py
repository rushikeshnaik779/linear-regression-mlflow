"""Preprocessing pipeline shared by every model variant."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler

from src.config import CAT_FEATURES, NUM_FEATURES


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
