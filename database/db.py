import sqlite3
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "fraud_detection.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            ml_fraud_probability REAL NOT NULL,
            dl_anomaly_score REAL NOT NULL,
            final_fraud_score REAL NOT NULL,
            status TEXT NOT NULL,
            shap_explanation TEXT,
            model_version TEXT,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(
    transaction_id,
    ml_fraud_probability,
    dl_anomaly_score,
    final_fraud_score,
    status,
    shap_explanation=None,
    model_version="1.0"
):
    connection = get_connection()

    connection.execute("""
        INSERT INTO predictions (
            transaction_id,
            ml_fraud_probability,
            dl_anomaly_score,
            final_fraud_score,
            status,
            shap_explanation,
            model_version,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        transaction_id,
        ml_fraud_probability,
        dl_anomaly_score,
        final_fraud_score,
        status,
        shap_explanation,
        model_version,
        datetime.utcnow().isoformat()
    ))

    connection.commit()
    connection.close()


def get_predictions(limit=100):
    connection = get_connection()

    rows = connection.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()

    connection.close()

    return [dict(row) for row in rows]