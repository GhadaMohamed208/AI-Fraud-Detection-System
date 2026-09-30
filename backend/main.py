import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from ensemble import HybridModel  # noqa: E402
from dl_pipeline import load_autoencoder, calculate_anomaly_score  # noqa: E402
from database.db import initialize_database, save_prediction  # noqa: E402
from explainability import Explainer  # noqa: E402

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test.csv"
XGB_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
META_MODEL_PATH = ARTIFACTS_DIR / "meta_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "anomaly_scaler.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessing_pipeline.pkl"
AUTOENCODER_PATH = MODELS_DIR / "autoencoder.pth"
CONFIG_PATH = ARTIFACTS_DIR / "hybrid_config.json"
RAW_FEATURE_NAMES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
PROCESSED_FEATURE_COUNT = 33
RAW_FEATURE_COUNT = len(RAW_FEATURE_NAMES)

app = FastAPI(
    title="AI Fraud Detection API",
    description="Hybrid AI Fraud Detection System with single-transaction and CSV batch analysis.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    xgb_model = joblib.load(XGB_PATH)
    meta_model = joblib.load(META_MODEL_PATH)
    anomaly_scaler = joblib.load(SCALER_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    autoencoder, ae_threshold, device = load_autoencoder(AUTOENCODER_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        hybrid_config = json.load(f)
    threshold = float(hybrid_config["threshold"])
    model_version = hybrid_config.get("model_version", "hybrid_v2")
    if len(preprocessor.feature_order) != PROCESSED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {PROCESSED_FEATURE_COUNT} processed features, "
            f"got {len(preprocessor.feature_order)}."
        )
    hybrid_model = HybridModel(xgb_model, meta_model, anomaly_scaler, threshold)
    explainer = Explainer(xgb_model, feature_names=preprocessor.feature_order)
except Exception as exc:
    print("ERROR LOADING MODELS:", exc)
    raise

initialize_database()


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=100)
    features: list[float]


class DemoPredictionRequest(BaseModel):
    sample_id: str = Field(..., min_length=1, max_length=50)


def _prepare_raw_transaction(features: list[float]) -> pd.DataFrame:
    values = np.asarray(features, dtype=float)
    if values.shape != (RAW_FEATURE_COUNT,):
        raise ValueError(
            f"Expected exactly {RAW_FEATURE_COUNT} raw features "
            f"({', '.join(RAW_FEATURE_NAMES)}), got {values.size}."
        )
    if not np.isfinite(values).all():
        raise ValueError("All feature values must be finite numbers.")
    if values[0] < 0:
        raise ValueError("Time cannot be negative.")
    if values[1] < 0:
        raise ValueError("Amount cannot be negative.")
    return pd.DataFrame([values], columns=RAW_FEATURE_NAMES)


def _safe_transaction_id(df: pd.DataFrame, row_index: int) -> str:
    for candidate in ("Transaction_ID", "transaction_id", "TransactionID", "ID", "id"):
        if candidate in df.columns and pd.notna(df.iloc[row_index][candidate]):
            return str(df.iloc[row_index][candidate])
    return f"CSV-{row_index + 1:06d}"


def _prepare_csv(df: pd.DataFrame):
    missing = [c for c in RAW_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required model columns: " + ", ".join(missing)
        )

    raw = df[RAW_FEATURE_NAMES].copy()
    for col in RAW_FEATURE_NAMES:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if raw.isna().any().any():
        bad = raw.columns[raw.isna().any()].tolist()
        raise ValueError(
            "CSV contains missing or non-numeric values in: " + ", ".join(bad)
        )
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("CSV contains non-finite numeric values.")
    if (raw["Time"] < 0).any():
        raise ValueError("Time cannot contain negative values.")
    if (raw["Amount"] < 0).any():
        raise ValueError("Amount cannot contain negative values.")
    return raw


def _shap_for_row(X_processed: pd.DataFrame, row_index: int, n=5):
    shap_df = explainer.top_features(X_processed.iloc[[row_index]], n=n)
    if shap_df is None or shap_df.empty:
        return []
    return [
        {"feature": str(r["feature"]), "shap_value": float(r["shap_value"])}
        for _, r in shap_df[["feature", "shap_value"]].iterrows()
    ]


def _predict_processed(transaction_id: str, X_processed: pd.DataFrame, save=True):
    anomaly_scores = calculate_anomaly_score(
        autoencoder, X_processed.to_numpy(dtype=np.float32), device
    )
    result = hybrid_model.predict(X_processed, anomaly_scores)
    ml_probability = float(np.asarray(result["fraud_probability"]).ravel()[0])
    dl_anomaly_score = float(np.asarray(result["anomaly_score"]).ravel()[0])
    final_fraud_score = float(np.asarray(result["final_fraud_score"]).ravel()[0])
    status = str(np.asarray(result["status"]).ravel()[0])

    shap_explanation = None
    if status == "Fraud":
        shap_explanation = _shap_for_row(X_processed, 0)

    if save:
        save_prediction(
            transaction_id=transaction_id,
            ml_fraud_probability=ml_probability,
            dl_anomaly_score=dl_anomaly_score,
            final_fraud_score=final_fraud_score,
            status=status,
            shap_explanation=json.dumps(shap_explanation),
            model_version=model_version,
        )

    return {
        "transaction_id": transaction_id,
        "ml_fraud_probability": ml_probability,
        "dl_anomaly_score": dl_anomaly_score,
        "final_fraud_score": final_fraud_score,
        "status": status,
        "shap_explanation": shap_explanation,
        "model_version": model_version,
        "threshold": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _load_demo_samples():
    if not TEST_DATA_PATH.exists():
        return {}
    df = pd.read_csv(TEST_DATA_PATH)
    required = set(preprocessor.feature_order) | {"Class"}
    if not required.issubset(df.columns):
        return {}
    samples = {}
    for sample_id, class_value in (("DEMO-001", 1), ("DEMO-002", 0)):
        subset = df[df["Class"] == class_value]
        if not subset.empty:
            row = subset.iloc[0]
            samples[sample_id] = {
                "row": row[preprocessor.feature_order].copy(),
                "class": int(class_value),
            }
    return samples


DEMO_SAMPLES = _load_demo_samples()


from fastapi.staticfiles import StaticFiles

# Serve the complete frontend (HTML, CSS, JS, images, etc.) from FastAPI.
FRONTEND_DIR = BASE_DIR / "frontend"

if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Frontend directory not found: {FRONTEND_DIR}")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)



@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True,
        "threshold": threshold,
        "raw_feature_count": RAW_FEATURE_COUNT,
        "processed_feature_count": PROCESSED_FEATURE_COUNT,
        "model_version": model_version,
        "demo_samples": len(DEMO_SAMPLES),
        "csv_analysis": True,
    }


@app.get("/demo/transactions")
def demo_transactions():
    if not DEMO_SAMPLES:
        raise HTTPException(status_code=503, detail="Demo transactions are unavailable.")
    choices = []
    for sample_id, sample in DEMO_SAMPLES.items():
        row = sample["row"]
        scaled_values = np.array([[row["Time"], row["Amount"], row["Amount_log"]]])
        raw_values = preprocessor.scaler.inverse_transform(scaled_values)[0]
        choices.append(
            {
                "sample_id": sample_id,
                "display_name": f"Sample Transaction {sample_id[-3:]}",
                "time": float(raw_values[0]),
                "amount": max(0.0, float(raw_values[1])),
                "known_label": "Fraud" if sample["class"] == 1 else "Legitimate",
            }
        )
    return {"transactions": choices}


@app.post("/predict")
def predict(transaction: TransactionRequest):
    try:
        X_raw = _prepare_raw_transaction(transaction.features)
        X_processed = preprocessor.transform(X_raw)
        return _predict_processed(transaction.transaction_id, X_processed)
    except HTTPException:
        raise
    except Exception as exc:
        print("Prediction error:", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/demo/predict")
def demo_predict(request: DemoPredictionRequest):
    sample = DEMO_SAMPLES.get(request.sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Demo transaction not found.")
    try:
        return _predict_processed(request.sample_id, sample["row"].to_frame().T)
    except Exception as exc:
        print("Demo prediction error:", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-csv")
def predict_csv(file: UploadFile = File(...)):
    """Analyze every transaction in an uploaded CSV.

    Required columns: Time, Amount, V1..V28.
    Optional columns: Transaction_ID/ID and Class.
    Class is never used as an input feature; when present it is used only for
    post-prediction comparison (Actual vs Predicted).
    """
    filename = file.filename or "uploaded.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        content = file.file.read()
        if not content:
            raise ValueError("The uploaded CSV is empty.")
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise ValueError("The uploaded CSV has no transaction rows.")
        if len(df) > 100000:
            raise ValueError("CSV is too large. Maximum supported size is 100,000 rows.")

        raw = _prepare_csv(df)
        X_processed = preprocessor.transform(raw)
        anomaly_scores = calculate_anomaly_score(
            autoencoder, X_processed.to_numpy(dtype=np.float32), device
        )
        result = hybrid_model.predict(X_processed, anomaly_scores)

        ml_probs = np.asarray(result["fraud_probability"]).ravel()
        normalized_anomaly = np.asarray(result["anomaly_score"]).ravel()
        final_scores = np.asarray(result["final_fraud_score"]).ravel()
        statuses = np.asarray(result["status"]).ravel()

        has_class = "Class" in df.columns
        actual = None
        if has_class:
            actual = pd.to_numeric(df["Class"], errors="coerce")
            if actual.isna().any() or not actual.isin([0, 1]).all():
                raise ValueError("Optional Class column must contain only 0 or 1 values.")
            actual = actual.astype(int).to_numpy()

        rows = []
        fraud_count = int(np.sum(statuses == "Fraud"))
        shap_by_row = {}
        fraud_indices = np.where(statuses == "Fraud")[0]
        if len(fraud_indices):
            fraud_shap = explainer.top_features_batch(X_processed.iloc[fraud_indices], n=5)
            shap_by_row = {int(idx): values for idx, values in zip(fraud_indices, fraud_shap)}

        for i in range(len(df)):
            status = str(statuses[i])
            shap_explanation = None
            top_reason = "No significant fraud indicators"
            if status == "Fraud":
                shap_explanation = shap_by_row.get(i, [])
                positive = [x for x in shap_explanation if x["shap_value"] > 0]
                reasons = positive if positive else shap_explanation
                top_reason = ", ".join(x["feature"] for x in reasons[:3]) or "Model anomaly pattern"

            row = {
                "transaction_id": _safe_transaction_id(df, i),
                "amount": float(raw.iloc[i]["Amount"]),
                "prediction": status,
                "fraud_score": float(final_scores[i]),
                "ml_probability": float(ml_probs[i]),
                "anomaly_score": float(normalized_anomaly[i]),
                "reason": top_reason,
                "shap_explanation": shap_explanation,
            }
            if has_class:
                row["actual_class"] = int(actual[i])
                row["actual_label"] = "Fraud" if actual[i] == 1 else "Legitimate"
                row["correct"] = (
                    (actual[i] == 1 and status == "Fraud")
                    or (actual[i] == 0 and status == "Legitimate")
                )
            rows.append(row)

            save_prediction(
                transaction_id=row["transaction_id"],
                ml_fraud_probability=float(ml_probs[i]),
                dl_anomaly_score=float(normalized_anomaly[i]),
                final_fraud_score=float(final_scores[i]),
                status=status,
                shap_explanation=json.dumps(shap_explanation),
                model_version=model_version,
            )

        summary = {
            "filename": filename,
            "total_transactions": len(rows),
            "fraud_count": fraud_count,
            "legitimate_count": len(rows) - fraud_count,
            "fraud_rate": fraud_count / len(rows),
            "threshold": threshold,
            "model_version": model_version,
            "has_actual_labels": has_class,
        }

        if has_class:
            predicted = np.array([1 if s == "Fraud" else 0 for s in statuses])
            summary["correct_predictions"] = int((predicted == actual).sum())
            summary["accuracy"] = float((predicted == actual).mean())
            summary["false_positives"] = int(((predicted == 1) & (actual == 0)).sum())
            summary["false_negatives"] = int(((predicted == 0) & (actual == 1)).sum())

        return {"summary": summary, "rows": rows}
    except HTTPException:
        raise
    except Exception as exc:
        print("CSV analysis error:", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Frontend assets are served last so API routes such as /predict-csv are matched first.
@app.get("/{path:path}")
def frontend_assets(path: str):
    requested = (FRONTEND_DIR / path).resolve()
    frontend_root = FRONTEND_DIR.resolve()
    if requested.is_file() and frontend_root in requested.parents:
        return FileResponse(requested)
    raise HTTPException(status_code=404, detail="Frontend asset not found.")
