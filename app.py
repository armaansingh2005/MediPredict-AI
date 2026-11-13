from __future__ import annotations

import io
from typing import Any, Iterable

import pandas as pd
from flask import Flask, jsonify, render_template, request

from disease_predictor.data import SYMPTOM_CATALOG
from disease_predictor.database import (
    clear_prediction_history,
    get_prediction_history,
    init_db,
    save_prediction_record,
)
from disease_predictor.model import ClinicalRiskModel, DatasetManager, list_saved_models, predict_disease


init_db()


def _normalize_user_symptoms(raw_symptoms: Any):
    """Normalize symptom input from a JSON request into a clean list."""
    if raw_symptoms is None:
        raise ValueError("Provide symptoms as a string, list, or selection object.")

    if isinstance(raw_symptoms, str):
        parts = raw_symptoms.replace(";", ",").split(",")
        cleaned = [part.strip().lower() for part in parts if part.strip()]
        if not cleaned:
            raise ValueError("At least one valid symptom is required.")
        return cleaned

    if isinstance(raw_symptoms, dict):
        if not raw_symptoms:
            raise ValueError("At least one valid symptom is required.")
        if any(not isinstance(name, str) for name in raw_symptoms):
            raise TypeError("Symptoms must contain only strings.")
        for name, selected in raw_symptoms.items():
            if not isinstance(selected, bool) and not (
                isinstance(selected, int) and selected in (0, 1)
            ):
                raise TypeError("Symptom selections must be boolean values.")
        selected = [
            str(name).strip().lower()
            for name, is_selected in raw_symptoms.items()
            if bool(is_selected) and str(name).strip()
        ]
        if not selected:
            raise ValueError("At least one valid symptom is required.")
        return selected

    if isinstance(raw_symptoms, Iterable) and not isinstance(raw_symptoms, (bytes, bytearray)):
        items = list(raw_symptoms)
        if not items:
            raise ValueError("At least one valid symptom is required.")
        normalized = []
        for item in items:
            if not isinstance(item, str):
                raise TypeError("Symptoms must contain only strings.")
            for chunk in item.replace(";", ",").split(","):
                cleaned = chunk.strip().lower()
                if cleaned:
                    normalized.append(cleaned)
        if not normalized:
            raise ValueError("At least one valid symptom is required.")
        return normalized

    raise TypeError("Provide symptoms as a string, list, or selection object.")


def _error_response(message: str, status_code: int = 400, **extras):
    payload = {"error": message}
    payload.update(extras)
    return jsonify(payload), status_code


def get_dashboard_summary() -> dict[str, int]:
    return {
        "symptoms": len(SYMPTOM_CATALOG),
        "history": len(get_prediction_history(limit=1000)),
        "datasets": len(DatasetManager.dataset_status()),
        "models": len(list_saved_models()),
    }


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            symptoms=SYMPTOM_CATALOG,
            disease_modules=[
                "diabetes",
                "heart",
                "parkinsons",
                "breast_cancer",
            ],
            dataset_sources=DatasetManager.dataset_status(),
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/datasets")
    def dataset_status():
        return jsonify(DatasetManager.dataset_status())

    @app.get("/api/model-status")
    def model_status():
        return jsonify({"models": list_saved_models()})

    @app.get("/api/overview")
    def overview():
        return jsonify(get_dashboard_summary())

    @app.get("/api/history")
    def prediction_history():
        return jsonify({"history": get_prediction_history(limit=20)})

    @app.delete("/api/history")
    def clear_history():
        deleted = clear_prediction_history()
        return jsonify({"status": "ok", "deleted": deleted})

    @app.post("/api/clinical-predict")
    def clinical_predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error_response("Request body must be a JSON object.")

        disease = str(data.get("disease", "")).strip().lower()
        if not disease:
            return _error_response("Disease module is required.")
        if "features" not in data:
            return _error_response("Features are required for clinical prediction.")

        try:
            service = ClinicalRiskModel()
            result = service.predict(disease, data["features"])
        except ValueError as exc:
            return _error_response(str(exc))
        except RuntimeError as exc:
            return _error_response(str(exc), status_code=503)
        except Exception as exc:  # pragma: no cover - defensive fallback
            return _error_response(f"Prediction failed: {exc}")

        return jsonify(result)

    @app.post("/api/upload-dataset")
    def upload_dataset():
        disease = ""
        uploaded = None

        if request.files and "file" in request.files:
            uploaded = request.files["file"]
            disease = str(request.form.get("disease", "")).strip().lower()
        elif request.is_json:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return _error_response("Request body must be a JSON object.")
            disease = str(payload.get("disease", "")).strip().lower()
            csv_text = payload.get("csv")
            if not isinstance(csv_text, str):
                return _error_response("CSV text is required for dataset upload.")
            uploaded = io.StringIO(csv_text)
        else:
            return _error_response("Upload a CSV file or JSON payload containing CSV text.")

        if not disease:
            disease = "diabetes"

        if uploaded is None:
            return _error_response("Dataset file is required.")

        try:
            if hasattr(uploaded, "read") and not isinstance(uploaded, io.StringIO):
                csv_bytes = uploaded.read()
                if not csv_bytes:
                    return _error_response("Uploaded CSV file is empty.")
                frame = pd.read_csv(io.BytesIO(csv_bytes))
            else:
                csv_text = uploaded.getvalue() if hasattr(uploaded, "getvalue") else str(uploaded)
                if not str(csv_text).strip():
                    return _error_response("Uploaded CSV file is empty.")
                frame = pd.read_csv(io.StringIO(str(csv_text)))

            prepared, target_col, feature_columns = DatasetManager.validate_uploaded_frame(frame, disease)
            model = ClinicalRiskModel()
            train_payload = model.train_from_frame(disease, prepared)
            train_payload.pop("model", None)
            train_payload["uploaded_rows"] = int(len(prepared))
            train_payload["target_column"] = target_col
            train_payload["feature_columns"] = feature_columns
            return jsonify(train_payload)
        except ValueError as exc:
            return _error_response(str(exc))
        except Exception as exc:  # pragma: no cover - defensive fallback
            return _error_response(f"Dataset upload failed: {exc}")

    @app.post("/api/predict")
    def predict_route():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error_response("Request body must be a JSON object.")

        raw_symptoms = data.get("symptoms")
        if raw_symptoms is None and "input" in data:
            raw_symptoms = data["input"]

        if raw_symptoms is None:
            return _error_response("Provide symptoms as a string, list, or selection object.")

        try:
            symptoms = _normalize_user_symptoms(raw_symptoms)
        except TypeError as exc:
            return _error_response(str(exc))
        except ValueError as exc:
            return _error_response(str(exc))

        try:
            result = predict_disease(symptoms)
        except ValueError as exc:
            unknown = [
                item.strip().lower()
                for item in symptoms
                if item.strip().lower() not in set(SYMPTOM_CATALOG)
            ]
            return _error_response(str(exc), unknown_symptoms=sorted(set(unknown)))

        saved = save_prediction_record(
            disease=str(result["disease"]),
            symptoms=symptoms,
            confidence=float(result["confidence"]),
            probabilities=result["probabilities"],
        )
        result["history_id"] = saved["id"]
        result["saved_to_db"] = True
        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
