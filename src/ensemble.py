import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


class HybridModel:
    def __init__(self, xgb_model, meta_model, anomaly_scaler, threshold=0.29):
        self.xgb_model = xgb_model
        self.meta_model = meta_model
        self.anomaly_scaler = anomaly_scaler
        self.threshold = threshold

    def predict_ml_probability(self, X):
        return self.xgb_model.predict_proba(X)[:, 1]

    def normalize_anomaly_score(self, anomaly_scores):
        return self.anomaly_scaler.transform(
            np.asarray(anomaly_scores).reshape(-1, 1)
        ).ravel()

    def predict(self, X, anomaly_scores):
        ml_probability = self.predict_ml_probability(X)

        normalized_anomaly = self.normalize_anomaly_score(
            anomaly_scores
        )

        meta_features = pd.DataFrame({
            "fraud_probability": ml_probability,
            "anomaly_score": normalized_anomaly
        })

        final_score = self.meta_model.predict_proba(
            meta_features
        )[:, 1]

        status = np.where(
            final_score >= self.threshold,
            "Fraud",
            "Legitimate"
        )

        return {
            "fraud_probability": ml_probability,
            "anomaly_score": normalized_anomaly,
            "final_fraud_score": final_score,
            "status": status
        }