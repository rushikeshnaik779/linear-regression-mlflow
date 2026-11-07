"""Generate a synthetic house-price dataset for linear regression experiments.

The true price has a few things plain linear regression cannot capture on its
own (an interaction between size and distance, and a neighbourhood effect),
plus some missing values, so there is room to improve the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path("data/houses.csv")


def make_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sqft = rng.normal(1500, 500, n).clip(400, 4000)
    bedrooms = rng.integers(1, 6, n)
    age_years = rng.uniform(0, 60, n)
    distance_km = rng.exponential(8, n).clip(0.2, 40)
    neighborhood = rng.choice(["downtown", "suburb", "rural"], n, p=[0.3, 0.5, 0.2])

    neighborhood_effect = (
        pd.Series(neighborhood)
        .map({"downtown": 60_000, "suburb": 20_000, "rural": 0})
        .to_numpy()
    )

    price = (
        120 * sqft
        + 8_000 * bedrooms
        - 1_500 * age_years
        - 30_000 * np.log1p(distance_km)  # nonlinear: big drop near the city, flattening out
        + neighborhood_effect
        + 0.05 * sqft * (40 - distance_km)  # size is worth more near the city
        + rng.normal(0, 25_000, n)
    )

    df = pd.DataFrame(
        {
            "sqft": sqft.round(0),
            "bedrooms": bedrooms,
            "age_years": age_years.round(1),
            "distance_km": distance_km.round(2),
            "neighborhood": neighborhood,
            "price": price.round(2),
        }
    )

    # Inject missing values so the pipeline has to handle them.
    for col, frac in [("sqft", 0.05), ("age_years", 0.03)]:
        df.loc[rng.random(n) < frac, col] = np.nan

    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isna().sum())
