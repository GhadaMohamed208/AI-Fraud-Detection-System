"""Integration tests for the finalized FastAPI inference contract."""
import os
import sys
import pytest
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import FraudPreprocessor  # noqa: E402
from backend.main import TransactionRequest, predict  # noqa: E402


def _raw_features_from_processed_row(row, preprocessor):
    scaled = np.array([[row["Time"], row["Amount"], row["Amount_log"]]], dtype=float)
    raw_time, raw_amount, _ = preprocessor.scaler.inverse_transform(scaled)[0]
    return [
        float(raw_time),
        float(raw_amount),
        *[float(row[f"V{i}"]) for i in range(1, 29)],
    ]


def test_api_accepts_30_raw_features_and_returns_contract():
    preprocessor = FraudPreprocessor.load(
        os.path.join(PROJECT_ROOT, "artifacts", "preprocessing_pipeline.pkl")
    )
    test_df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    )
    row = test_df.iloc[0]
    features = _raw_features_from_processed_row(row, preprocessor)

    response = predict(
        TransactionRequest(
            transaction_id="PYTEST-API-001",
            features=features,
        )
    )

    assert len(features) == 30
    assert response["status"] in {"Fraud", "Legitimate"}
    assert 0 <= response["ml_fraud_probability"] <= 1
    assert 0 <= response["final_fraud_score"] <= 1
    assert response["threshold"] == pytest.approx(0.29)


def test_shap_is_only_returned_for_fraud():
    preprocessor = FraudPreprocessor.load(
        os.path.join(PROJECT_ROOT, "artifacts", "preprocessing_pipeline.pkl")
    )
    test_df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    )

    # Use a known fraud row so this test verifies the Fraud -> SHAP branch.
    fraud_row = test_df[test_df["Class"] == 1].iloc[0]
    fraud_features = _raw_features_from_processed_row(fraud_row, preprocessor)
    fraud_response = predict(
        TransactionRequest(
            transaction_id="PYTEST-FRAUD-001",
            features=fraud_features,
        )
    )

    assert fraud_response["status"] == "Fraud"
    assert isinstance(fraud_response["shap_explanation"], list)
    assert len(fraud_response["shap_explanation"]) > 0

    # Use a known legitimate row to verify that SHAP is omitted.
    legit_row = test_df[test_df["Class"] == 0].iloc[0]
    legit_features = _raw_features_from_processed_row(legit_row, preprocessor)
    legit_response = predict(
        TransactionRequest(
            transaction_id="PYTEST-LEGIT-001",
            features=legit_features,
        )
    )

    assert legit_response["status"] == "Legitimate"
    assert legit_response["shap_explanation"] is None
