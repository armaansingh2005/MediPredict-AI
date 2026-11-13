from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DISEASE_SYMPTOM_MAP, SYMPTOM_CATALOG

RANDOM_STATE = 42
TRAINING_SAMPLES_PER_DISEASE = 100
EXPECTED_SYMPTOM_RATE = 0.85
INCIDENTAL_SYMPTOM_RATE = 0.04
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def build_training_data() -> pd.DataFrame:
    """Create reproducible synthetic records with missing and incidental symptoms."""
    rng = random.Random(RANDOM_STATE)
    records: List[Dict[str, Any]] = []

    for disease, disease_symptoms in DISEASE_SYMPTOM_MAP.items():
        disease_symptom_set = set(disease_symptoms)

        for _ in range(TRAINING_SAMPLES_PER_DISEASE):
            record: Dict[str, Any] = {
                symptom: int(
                    rng.random()
                    < (
                        EXPECTED_SYMPTOM_RATE
                        if symptom in disease_symptom_set
                        else INCIDENTAL_SYMPTOM_RATE
                    )
                )
                for symptom in SYMPTOM_CATALOG
            }
            record["disease"] = disease
            records.append(record)

    return pd.DataFrame(records, columns=["disease", *SYMPTOM_CATALOG])


class DiseasePredictor:
    """Train and run a multi-class model over the project's symptom catalog."""

    def __init__(self) -> None:
        self.symptoms = list(SYMPTOM_CATALOG)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.3,
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        )
        self._fit()

    def _fit(self) -> None:
        """Fit the classifier on the deterministic synthetic training records."""
        training_data = build_training_data()
        self.model.fit(
            training_data[self.symptoms],
            training_data["disease"],
        )

    def _normalize_symptoms(self, symptoms: Any) -> List[str]:
        """Normalize supported string, iterable, and checkbox-map inputs."""
        if symptoms is None:
            raise ValueError("No symptoms were provided.")

        if isinstance(symptoms, str):
            parts = symptoms.replace(";", ",").split(",")
            return [part.strip().lower() for part in parts if part.strip()]

        if isinstance(symptoms, dict):
            if not all(isinstance(name, str) for name in symptoms):
                raise TypeError("Symptom names must be strings.")
            return [
                name.strip().lower()
                for name, selected in symptoms.items()
                if selected and name.strip()
            ]

        if isinstance(symptoms, Iterable) and not isinstance(
            symptoms, (bytes, bytearray)
        ):
            normalized: List[str] = []
            for item in symptoms:
                if not isinstance(item, str):
                    raise TypeError("Each symptom must be a string.")
                parts = item.replace(";", ",").split(",")
                normalized.extend(
                    part.strip().lower() for part in parts if part.strip()
                )
            return normalized

        raise TypeError("Symptoms must be a list, string, or dictionary.")

    def _vectorize(self, symptoms: Sequence[str]) -> List[int]:
        """Convert normalized symptom names into the model's fixed feature order."""
        if not all(isinstance(symptom, str) for symptom in symptoms):
            raise TypeError("Each symptom must be a string.")

        cleaned = [symptom.strip().lower() for symptom in symptoms if symptom.strip()]
        if not cleaned:
            raise ValueError("At least one valid symptom is required.")

        unknown_symptoms = sorted(set(cleaned) - set(self.symptoms))
        if unknown_symptoms:
            raise ValueError(f"Unknown symptoms: {', '.join(unknown_symptoms)}")

        selected_symptoms = set(cleaned)
        return [int(symptom in selected_symptoms) for symptom in self.symptoms]

    def _feature_frame(self, symptoms: Any) -> pd.DataFrame:
        """Normalize, validate, and order input features for scikit-learn."""
        normalized = self._normalize_symptoms(symptoms)
        vector = self._vectorize(normalized)
        return pd.DataFrame([vector], columns=self.symptoms)

    def predict(self, symptoms: Any) -> str:
        """Return the most likely class for the supplied symptoms."""
        feature_frame = self._feature_frame(symptoms)
        prediction = self.model.predict(feature_frame)[0]
        return str(prediction)

    def predict_proba(self, symptoms: Any) -> Dict[str, float]:
        """Return a probability for each disease class."""
        feature_frame = self._feature_frame(symptoms)
        probabilities = self.model.predict_proba(feature_frame)[0]
        labels = self.model.classes_
        return {
            str(label): float(probability)
            for label, probability in zip(labels, probabilities)
        }

    def predict_with_confidence(self, symptoms: Any) -> Dict[str, Any]:
        """Return the top class, its model score, and all class probabilities."""
        probabilities = self.predict_proba(symptoms)
        disease = max(probabilities, key=probabilities.__getitem__)
        confidence = probabilities[disease]
        return {
            "disease": disease,
            "confidence": round(float(confidence), 4),
            "probabilities": probabilities,
        }


symptom_instance = DiseasePredictor()


def predict_disease(symptoms: Any) -> Dict[str, Any]:
    """Predict with the shared, pre-trained symptom model instance."""
    return symptom_instance.predict_with_confidence(symptoms)


def create_predictor() -> DiseasePredictor:
    """Create an independent predictor with the same deterministic training data."""
    return DiseasePredictor()


class DatasetManager:
    """Manage public datasets for the structured medical prediction modules."""

    DATASET_SOURCES = {
        "diabetes": "sklearn.load_diabetes",
        "breast_cancer": "sklearn.load_breast_cancer",
        "heart": "OpenML heart disease dataset",
        "parkinsons": "OpenML Parkinsons dataset",
    }

    TARGET_ALIASES = {
        "diabetes": ["target", "outcome", "diabetes"],
        "heart": ["target", "class", "heart_disease"],
        "parkinsons": ["target", "status", "class", "parkinson"],
        "breast_cancer": ["target", "diagnosis", "label", "malignant"],
    }

    @staticmethod
    def normalize_disease_name(disease: str) -> str:
        return disease.strip().lower().replace(" ", "_")

    @staticmethod
    def _safe_fetch_openml(name: str):
        try:
            return fetch_openml(name=name, version=1, as_frame=True, parser="auto")
        except Exception:
            return None

    @staticmethod
    def _coerce_binary_target(series: pd.Series, disease: str) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        if values.notna().sum() == 0:
            labels = series.dropna().tolist()
            mapping = {label: idx for idx, label in enumerate(sorted(set(labels))) }
            return pd.Series(series.map(mapping), index=series.index).astype(int)

        if values.nunique() <= 2:
            unique_values = sorted(values.dropna().unique().tolist())
            if set(unique_values).issubset({0, 1}):
                return values.fillna(unique_values[0]).astype(int)
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            return values.map(mapping).astype(int)

        median = float(values.median())
        return (values >= median).astype(int)

    @staticmethod
    def _resolve_target_column(frame: pd.DataFrame, disease: str) -> str:
        lowered = {str(name).strip().lower(): name for name in frame.columns}
        aliases = DatasetManager.TARGET_ALIASES.get(disease, ["target"])
        for alias in aliases:
            if alias in lowered:
                return lowered[alias]
        for column in frame.columns:
            if "target" in str(column).lower() or "label" in str(column).lower() or "class" in str(column).lower():
                return column
        return frame.columns[-1]

    @staticmethod
    def validate_uploaded_frame(frame: pd.DataFrame, disease: str) -> tuple[pd.DataFrame, str, list[str]]:
        disease_key = DatasetManager.normalize_disease_name(disease)
        target_col = DatasetManager._resolve_target_column(frame, disease_key)
        prepared = frame.copy()
        prepared[target_col] = DatasetManager._coerce_binary_target(prepared[target_col], disease_key)

        feature_columns = [
            col for col in prepared.columns if col != target_col
        ]
        if not feature_columns:
            raise ValueError("Uploaded dataset must include at least one feature column.")

        if prepared[target_col].nunique() < 2:
            raise ValueError("Uploaded dataset must include at least two target classes for training.")

        feature_frame = prepared[feature_columns].copy()
        for column in feature_frame.columns:
            if pd.api.types.is_object_dtype(feature_frame[column]):
                feature_frame[column] = feature_frame[column].astype(str)
        feature_frame = pd.get_dummies(feature_frame, drop_first=False)
        prepared = pd.concat([feature_frame, prepared[target_col]], axis=1)
        return prepared, target_col, list(feature_frame.columns)

    @staticmethod
    def load_dataset(disease: str) -> tuple[pd.DataFrame, str, str]:
        disease_key = DatasetManager.normalize_disease_name(disease)
        if disease_key == "diabetes":
            data = load_diabetes(as_frame=True)
            frame = data.frame.iloc[:, :8].copy()
            frame.columns = [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ]
            frame["target"] = data.target
            frame["target"] = DatasetManager._coerce_binary_target(frame["target"], "diabetes")
            return frame, "target", "diabetes"
        if disease_key == "breast_cancer":
            data = load_breast_cancer(as_frame=True)
            frame = data.frame.copy()
            frame.columns = [
                "mean radius",
                "mean texture",
                "mean perimeter",
                "mean area",
                "mean smoothness",
                "mean compactness",
                "mean concavity",
                "mean concave points",
                "mean symmetry",
                "mean fractal dimension",
            ]
            frame["target"] = data.target
            frame["target"] = DatasetManager._coerce_binary_target(frame["target"], "breast_cancer")
            return frame, "target", "breast_cancer"
        if disease_key == "heart":
            dataset = DatasetManager._safe_fetch_openml("heart")
            if dataset is None:
                raise RuntimeError(
                    "Heart disease dataset could not be loaded automatically. "
                    "Please download the UCI Heart Disease dataset and place it in the project data folder."
                )
            frame = dataset.frame.copy()
            y_name = DatasetManager._resolve_target_column(frame, "heart")
            frame = frame.rename(columns={
                list(frame.columns)[0]: "Age",
                list(frame.columns)[1]: "Sex",
                list(frame.columns)[2]: "ChestPainType",
                list(frame.columns)[3]: "RestingBP",
                list(frame.columns)[4]: "Cholesterol",
                list(frame.columns)[5]: "FastingBS",
                list(frame.columns)[6]: "RestingECG",
                list(frame.columns)[7]: "MaxHR",
                list(frame.columns)[8]: "ExerciseAngina",
                list(frame.columns)[9]: "Oldpeak",
                list(frame.columns)[10]: "ST_Slope",
                list(frame.columns)[11]: "Ca",
                list(frame.columns)[12]: "Thal",
            })
            frame[y_name] = DatasetManager._coerce_binary_target(frame[y_name], "heart")
            return frame, y_name, "heart"
        if disease_key in {"parkinsons", "parkinson"}:
            dataset = DatasetManager._safe_fetch_openml("parkinsons")
            if dataset is None:
                raise RuntimeError(
                    "Parkinson's dataset could not be loaded automatically. "
                    "Please download the UCI Parkinson's dataset and place it in the project data folder."
                )
            frame = dataset.frame.copy()
            y_name = DatasetManager._resolve_target_column(frame, "parkinsons")
            feature_columns = [
                "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "Jitter(%)", "Jitter(Abs)", "Shimmer",
                "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
            ]
            for idx, name in enumerate(feature_columns):
                if idx < len(frame.columns) - 1:
                    frame = frame.rename(columns={frame.columns[idx]: name})
            frame[y_name] = DatasetManager._coerce_binary_target(frame[y_name], "parkinsons")
            return frame, y_name, "parkinsons"
        raise ValueError(f"Unsupported disease module: {disease}")

    @staticmethod
    def dataset_status() -> Dict[str, str]:
        return {key: value for key, value in DatasetManager.DATASET_SOURCES.items()}


class ClinicalRiskModel:
    """Train and persist a single disease-model pipeline."""

    VALID_TARGET_NAMES = {
        "diabetes": ["target", "outcome", "diabetes"],
        "heart": ["target", "class", "heart_disease"],
        "parkinsons": ["target", "status", "class", "parkinson"],
        "breast_cancer": ["target", "diagnosis", "label", "malignant"],
    }

    FEATURE_SETS = {
        "diabetes": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        "heart": [
            "Age",
            "Sex",
            "ChestPainType",
            "RestingBP",
            "Cholesterol",
            "FastingBS",
            "RestingECG",
            "MaxHR",
            "ExerciseAngina",
            "Oldpeak",
            "ST_Slope",
            "Ca",
            "Thal",
        ],
        "parkinsons": [
            "MDVP:Fo(Hz)",
            "MDVP:Fhi(Hz)",
            "MDVP:Flo(Hz)",
            "Jitter(%)",
            "Jitter(Abs)",
            "Jitter:RAP",
            "Jitter:PPQ",
            "Shimmer",
            "Shimmer(dB)",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "spread1",
            "spread2",
            "D2",
            "PPE",
        ],
        "breast_cancer": [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "mean compactness",
            "mean concavity",
            "mean concave points",
            "mean symmetry",
            "mean fractal dimension",
        ],
    }

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        )

    def _model_path(self, disease: str) -> Path:
        return self.model_dir / f"{disease.lower().replace(' ', '_')}_model.joblib"

    @staticmethod
    def _prepare_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        raw = frame[feature_columns].copy()
        for column in raw.columns:
            if pd.api.types.is_object_dtype(raw[column]):
                raw[column] = raw[column].astype(str)
        return pd.get_dummies(raw, drop_first=False)

    def train_from_frame(self, disease: str, frame: pd.DataFrame) -> Dict[str, Any]:
        disease_name = DatasetManager.normalize_disease_name(disease)
        target_col = DatasetManager._resolve_target_column(frame, disease_name)
        feature_candidates = [
            col for col in frame.columns if col != target_col and col in self.FEATURE_SETS.get(disease_name, [])
        ]
        if not feature_candidates:
            feature_candidates = [col for col in frame.columns if col != target_col]

        feature_columns = feature_candidates
        prepared = frame.copy()
        prepared[target_col] = DatasetManager._coerce_binary_target(prepared[target_col], disease_name)
        X = self._prepare_feature_matrix(prepared, feature_columns)
        y = prepared[target_col].astype(int)

        if y.nunique() < 2:
            raise ValueError(f"Dataset for {disease_name} does not contain enough target classes for training.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        model_path = self._model_path(disease_name)
        model_payload = {
            "disease": disease_name,
            "feature_columns": list(X.columns),
            "target_column": target_col,
            "model": pipeline,
            "accuracy": float(accuracy),
            "source": DatasetManager.DATASET_SOURCES.get(disease_name, "manual"),
            "dataset_columns": list(prepared.columns),
        }
        import joblib

        joblib.dump(model_payload, model_path)
        return model_payload

    def train(self, disease: str) -> Dict[str, Any]:
        frame, target_col, disease_name = DatasetManager.load_dataset(disease)
        return self.train_from_frame(disease_name, frame)

    def load(self, disease: str) -> Dict[str, Any]:
        disease_key = DatasetManager.normalize_disease_name(disease)
        model_path = self._model_path(disease_key)
        if model_path.exists():
            import joblib

            return joblib.load(model_path)
        return self.train(disease_key)

    def predict(self, disease: str, feature_values: Dict[str, Any] | Sequence[float]) -> Dict[str, Any]:
        payload = self.load(disease)
        model = payload["model"]
        feature_columns = payload["feature_columns"]

        if isinstance(feature_values, dict):
            normalized = {}
            for key, value in feature_values.items():
                normalized[str(key).strip()] = value
            values = []
            for feature in feature_columns:
                candidate_keys = [feature, feature.lower(), feature.replace(" ", "_"), feature.replace(" ", "").lower()]
                matched = None
                for key in candidate_keys:
                    if key in normalized:
                        matched = key
                        break
                if matched is None:
                    if feature not in normalized:
                        raise ValueError(f"Missing required value for feature: {feature}")
                    matched = feature
                values.append(float(normalized[matched]))
        else:
            values = [float(value) for value in feature_values]
            if len(values) != len(feature_columns):
                raise ValueError(
                    f"Feature count mismatch. Expected {len(feature_columns)} values, received {len(values)}."
                )

        frame = pd.DataFrame([values], columns=feature_columns)
        probabilities = model.predict_proba(frame)[0]
        labels = model.classes_
        index = int(np.argmax(probabilities))
        return {
            "disease": disease,
            "label": str(labels[index]),
            "confidence": float(probabilities[index]),
            "probabilities": {str(label): float(prob) for label, prob in zip(labels, probabilities)},
            "model_accuracy": float(payload.get("accuracy", 0.0)),
        }


clinical_instance = ClinicalRiskModel()


def create_clinical_predictor() -> ClinicalRiskModel:
    return ClinicalRiskModel()


def list_saved_models(model_dir: str | Path = MODEL_DIR) -> List[Dict[str, Any]]:
    """Return summary metadata for every persisted clinical model."""
    directory = Path(model_dir)
    results: List[Dict[str, Any]] = []
    if not directory.exists():
        return results

    import joblib

    for model_path in sorted(directory.glob("*_model.joblib")):
        try:
            payload = joblib.load(model_path)
        except Exception:
            continue

        disease_name = str(payload.get("disease") or model_path.stem.replace("_model", "")).strip()
        results.append(
            {
                "disease": disease_name,
                "accuracy": float(payload.get("accuracy", 0.0)),
                "source": str(payload.get("source", "manual")),
                "feature_columns": list(payload.get("feature_columns", [])),
                "target_column": str(payload.get("target_column", "target")),
                "model_file": model_path.name,
                "updated_at": model_path.stat().st_mtime,
            }
        )

    return results
