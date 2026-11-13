# MediPredict AI – Disease Prediction System

This project is a full-stack disease prediction web application designed to combine an interactive symptom-based classifier with structured clinical prediction modules for multi-disease modeling. It exposes both a lightweight symptom API and a clinical prediction endpoint, and it includes support for real public datasets where available.

## Features

- Symptom-driven disease prediction with confidence scores
- Clinical prediction module for diabetes, heart disease, Parkinson's disease, and breast cancer
- Real public dataset integration through scikit-learn and OpenML loaders
- Data-source registry and model metadata for each disease module
- SQLite-ready application architecture and joblib model persistence
- Responsive Flask UI with prediction cards and disease-specific numerical input forms
- JSON endpoints and health checks for application integration

## Model notes

The project keeps a deterministic symptom classifier trained on synthetic but structured symptom profiles to preserve the demo behavior and legacy API compatibility. The clinical modules are implemented to load real datasets when available and train logistic-regression pipelines with proper preprocessing.

This project is an educational and research-oriented application. It is not a substitute for professional medical advice and should never be used for direct clinical decision-making without expert review and validation.

## Quick start

```powershell
cd "C:\Users\Seagrey\Desktop\Armaan docs\ML Projects\Disease Prediction System"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000/ in the browser.

## API

- `GET /health` returns `{"status": "ok"}`.
- `POST /api/predict` accepts `symptoms` as a list, string, or boolean mapping. Legacy `input` is also accepted.
- `POST /api/clinical-predict` accepts `{"disease": "diabetes", "features": {...}}` and returns a probability-based prediction.
- `GET /api/datasets` exposes dataset source metadata for each disease module.

## Example symptoms

Try these symptoms to test the legacy symptom classifier:

- fever, cough, shortness_of_breath, chest_pain, fatigue -> Pneumonia
- fever, headache, body_ache, rash, nausea, joint_pain -> Dengue
- runny_nose, sneezing, sore_throat, mild_fever -> Common Cold

## Dataset sources

The project intends to use public datasets such as:

- Diabetes: scikit-learn `load_diabetes`
- Breast cancer: scikit-learn `load_breast_cancer`
- Heart disease: OpenML heart disease dataset (or a validated local CSV upload)
- Parkinson's disease: OpenML Parkinsons dataset (or a validated local CSV upload)

If automatic access to the remote dataset is unavailable, the API returns a clear error and asks the user to place a local dataset file in the project folder or provide a dataset manually.

## Project structure

```text
Disease Prediction System/
├── app.py
├── requirements.txt
├── README.md
├── disease_predictor/
│   ├── __init__.py
│   ├── app.py
│   ├── data.py
│   └── model.py
├── templates/
│   └── index.html
├── tests/
│   └── test_predictor.py
└── .gitignore
```
