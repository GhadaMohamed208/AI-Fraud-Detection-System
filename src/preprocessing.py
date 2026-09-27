"""
Reusable, leakage-safe preprocessing pipeline for the credit card fraud dataset.

Usage:
    pre = FraudPreprocessor(random_state=42)
    pre.fit(X_train_raw)                    # fit ONLY on training data
    X_train_final = pre.transform(X_train_raw)
    X_val_final   = pre.transform(X_val_raw)
    X_test_final  = pre.transform(X_test_raw)

    pre.save("artifacts/preprocessing_pipeline.pkl")
    pre2 = FraudPreprocessor.load("artifacts/preprocessing_pipeline.pkl")
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from features import engineer_features, ENGINEERED_FEATURE_NAMES


class FraudPreprocessor:
    """Deterministic, reusable preprocessing pipeline.

    Fit only on training data. transform() is safe to call on validation, test,
    or a single new transaction at inference time.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_order = None  # fixed at fit time
        self._scale_cols = None
        self._is_fit = False

    def _build_raw_feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering (Section 9) to a raw input frame."""
        return engineer_features(X)

    def fit(self, X: pd.DataFrame):
        X_eng = self._build_raw_feature_frame(X)
        scale_cols = ["Time", "Amount", "Amount_log"]
        self._scale_cols = scale_cols
        self.scaler.fit(X_eng[scale_cols])

        # Fixed feature order: scaled columns first (in a stable order), then PCA columns,
        # then remaining engineered (already-bounded) columns.
        pca_cols = [f"V{i}" for i in range(1, 29)]
        other_engineered = [c for c in X_eng.columns if c not in scale_cols + pca_cols]
        self.feature_order = scale_cols + pca_cols + sorted(other_engineered)
        self._is_fit = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError("FraudPreprocessor.transform() called before fit().")
        X_eng = self._build_raw_feature_frame(X)
        X_eng = X_eng.copy()
        X_eng[self._scale_cols] = self.scaler.transform(X_eng[self._scale_cols])
        # Enforce fixed column order; raises if a downstream user drops/renames a column.
        return X_eng[self.feature_order]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FraudPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj
