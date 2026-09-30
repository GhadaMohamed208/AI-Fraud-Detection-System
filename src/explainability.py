import numpy as np
import pandas as pd
import xgboost as xgb


class Explainer:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.booster = model.get_booster()

    def top_features_batch(self, X, n=5):
        """Return top SHAP contributors for every row in X."""
        if hasattr(X, "toarray"):
            X_array = X.toarray()
        else:
            X_array = np.asarray(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        feature_names = list(self.feature_names) if self.feature_names is not None else [
            f"feature_{i}" for i in range(X_array.shape[1])
        ]
        dmatrix = xgb.DMatrix(X_array, feature_names=feature_names)
        shap_values = self.booster.predict(dmatrix, pred_contribs=True)
        feature_values = shap_values[:, :-1]
        top_indices = np.argsort(np.abs(feature_values), axis=1)[:, ::-1][:, :n]

        results = []
        for row_idx in range(X_array.shape[0]):
            row = []
            for feature_idx in top_indices[row_idx]:
                row.append({
                    "feature": feature_names[int(feature_idx)],
                    "shap_value": float(feature_values[row_idx, feature_idx]),
                })
            results.append(row)
        return results

    def top_features(self, X, n=5):
        batch = self.top_features_batch(X, n=n)
        return pd.DataFrame(batch[0]) if batch and batch[0] else pd.DataFrame(columns=["feature", "shap_value"])
