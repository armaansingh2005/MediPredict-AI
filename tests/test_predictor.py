import io
import random

import pandas as pd
import pytest
from sklearn.metrics import f1_score

from disease_predictor import DiseasePredictor, create_app, predict_disease
from disease_predictor.data import DISEASE_SYMPTOM_MAP, SYMPTOM_CATALOG
from disease_predictor.model import ClinicalRiskModel, DatasetManager, build_training_data


def test_known_disease_prediction():
    predictor = DiseasePredictor()
    diagnosis = predictor.predict(["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue"])
    assert diagnosis == "Pneumonia"


def test_api_helper_returns_disease_and_confidence():
    result = predict_disease(["runny_nose", "sneezing", "sore_throat", "mild_fever"])
    assert "disease" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["disease"] == "Common Cold"


def test_training_data_is_repeatable_without_mutating_global_random_state():
    random.seed(123)
    random_state = random.getstate()

    first = build_training_data()
    second = build_training_data()

    assert random.getstate() == random_state
    pd.testing.assert_frame_equal(first, second)


def test_model_generalizes_to_missing_and_incidental_symptoms():
    predictor = DiseasePredictor()
    rng = random.Random(2026)
    records = []
    labels = []

    for disease, expected_symptoms in DISEASE_SYMPTOM_MAP.items():
        expected_symptoms = set(expected_symptoms)
        for _ in range(100):
            records.append(
                [
                    int(
                        rng.random()
                        < (0.85 if symptom in expected_symptoms else 0.04)
                    )
                    for symptom in predictor.symptoms
                ]
            )
            labels.append(disease)

    validation_data = pd.DataFrame(records, columns=predictor.symptoms)
    predictions = predictor.model.predict(validation_data)

    assert f1_score(labels, predictions, average="macro") >= 0.95


def test_predictor_rejects_unknown_and_non_string_symptoms():
    predictor = DiseasePredictor()

    with pytest.raises(ValueError, match="Unknown symptoms"):
        predictor.predict(["fever", "not_a_symptom"])

    with pytest.raises(TypeError, match="Each symptom must be a string"):
        predictor.predict(["fever", 1])


def test_clinical_model_trains_builtin_diabetes_dataset():
    payload = ClinicalRiskModel().train("diabetes")

    assert payload["disease"] == "diabetes"
    assert payload["target_column"] == "target"
    assert len(payload["feature_columns"]) >= 2
    assert 0.0 <= payload["accuracy"] <= 1.0


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_homepage_and_health_endpoint(client):
    homepage = client.get("/")
    health = client.get("/health")

    assert homepage.status_code == 200
    assert b"Select the symptoms" in homepage.data
    assert health.status_code == 200
    assert health.get_json() == {"status": "ok"}


def test_dashboard_overview_endpoint(client):
    response = client.get("/api/overview")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["symptoms"] == len(SYMPTOM_CATALOG)
    assert payload["datasets"] == len(DatasetManager.dataset_status())
    assert "history" in payload
    assert "models" in payload


def test_history_clear_endpoint(client):
    client.post("/api/predict", json={"symptoms": ["fever", "cough"]})
    response = client.delete("/api/history")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["deleted"] >= 1

    history_response = client.get("/api/history")
    assert history_response.status_code == 200
    assert history_response.get_json()["history"] == []


@pytest.mark.parametrize(("disease", "symptoms"), DISEASE_SYMPTOM_MAP.items())
def test_prediction_api_returns_each_disease(client, disease, symptoms):
    response = client.post("/api/predict", json={"symptoms": symptoms})

    assert response.status_code == 200
    result = response.get_json()
    assert result["disease"] == disease
    assert 0 <= result["confidence"] <= 1
    assert sum(result["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        (None, "Request body must be a JSON object."),
        ([], "Request body must be a JSON object."),
        ({}, "Provide symptoms as a string, list, or selection object."),
        ({"symptoms": []}, "At least one valid symptom is required."),
        ({"symptoms": ["fever", 1]}, "Symptoms must contain only strings."),
        (
            {"symptoms": {"fever": "yes"}},
            "Symptom selections must be boolean values.",
        ),
        (
            {"symptoms": {"fever": 1.0}},
            "Symptom selections must be boolean values.",
        ),
    ],
)
def test_prediction_api_rejects_invalid_json_payloads(client, body, expected_error):
    response = client.post("/api/predict", json=body)

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json()["error"] == expected_error


def test_prediction_api_rejects_malformed_json(client):
    response = client.post(
        "/api/predict", data="{", content_type="application/json"
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Request body must be a JSON object."}

    response = client.post(
        "/api/predict", data='{"symptoms":["fever"]}', content_type="text/plain"
    )

    assert response.status_code == 400
    assert response.is_json


def test_prediction_api_rejects_unknown_symptoms(client):
    response = client.post(
        "/api/predict", json={"symptoms": ["fever", "not_a_symptom"]}
    )

    assert response.status_code == 400
    assert response.get_json()["unknown_symptoms"] == ["not_a_symptom"]


def test_prediction_api_preserves_input_alias_and_string_support(client):
    response = client.post(
        "/api/predict", json={"input": "runny_nose; sneezing, sore_throat, mild_fever"}
    )

    assert response.status_code == 200
    assert response.get_json()["disease"] == "Common Cold"


def test_prediction_api_accepts_selection_mapping(client):
    response = client.post(
        "/api/predict",
        json={
            "symptoms": {
                "runny_nose": True,
                "sneezing": 1,
                "sore_throat": True,
                "mild_fever": 1,
                "fever": False,
            }
        },
    )

    assert response.status_code == 200
    assert response.get_json()["disease"] == "Common Cold"


def test_dataset_upload_api_accepts_csv(client):
    csv_rows = [
        "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,target",
    ]
    for _ in range(12):
        csv_rows.append(
            "1,85,66,29,0,26.6,0.351,31,0"
        )
    for _ in range(12):
        csv_rows.append(
            "5,166,72,19,175,25.8,0.587,51,1"
        )
    csv_text = "\n".join(csv_rows)

    response = client.post(
        "/api/upload-dataset",
        data={
            "disease": "diabetes",
            "file": (io.BytesIO(csv_text.encode("utf-8")), "diabetes.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["disease"] == "diabetes"
    assert payload["uploaded_rows"] >= 20
    assert "accuracy" in payload
