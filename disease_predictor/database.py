from __future__ import annotations

import ast
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path(__file__).resolve().parent.parent / "medi_predict.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction_record(disease: str, symptoms: List[str], confidence: float, probabilities: Dict[str, float]) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {"disease": disease, "symptoms": ",".join(symptoms), "confidence": float(confidence), "probabilities": str(probabilities), "created_at": timestamp}
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_history (disease, symptoms, confidence, probabilities, created_at)
            VALUES (:disease, :symptoms, :confidence, :probabilities, :created_at)
            """,
            payload,
        )
        conn.commit()
        row_id = cursor.lastrowid

    return {
        "id": row_id,
        "disease": disease,
        "symptoms": symptoms,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "created_at": timestamp,
    }


def get_prediction_history(limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, disease, symptoms, confidence, probabilities, created_at
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history = []
    for row in rows:
        probs = row["probabilities"]
        try:
            probabilities = ast.literal_eval(probs) if isinstance(probs, str) else probs
        except (ValueError, SyntaxError):
            probabilities = {}

        history.append(
            {
                "id": row["id"],
                "disease": row["disease"],
                "symptoms": [item.strip() for item in row["symptoms"].split(",") if item.strip()],
                "confidence": float(row["confidence"]),
                "probabilities": probabilities if isinstance(probabilities, dict) else {},
                "created_at": row["created_at"],
            }
        )
    return history


def clear_prediction_history() -> int:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM prediction_history")
        conn.commit()
        return int(cursor.rowcount)
