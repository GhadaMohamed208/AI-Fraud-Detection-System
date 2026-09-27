"""
Row-wise, inference-safe feature engineering for the credit card fraud dataset.
Every feature here is computable from a single raw transaction alone.
"""
import numpy as np
import pandas as pd

ENGINEERED_FEATURE_NAMES = ["Hour_sin", "Hour_cos", "Amount_log"]


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with engineered columns added. Does not mutate the input."""
    X = X.copy()

    hour_of_day = (X["Time"] // 3600) % 24
    X["Hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    X["Hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)

    X["Amount_log"] = np.log1p(X["Amount"])

    return X
