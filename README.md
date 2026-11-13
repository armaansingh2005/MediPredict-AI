# 🏥 MediPredict AI – Disease Prediction System

### An AI/ML-Powered Healthcare Risk Prediction Platform

MediPredict AI is a machine learning-based disease prediction web application designed to estimate potential disease risks using user-provided symptoms and clinical health information. It combines symptom-based prediction with clinical prediction modules through an interactive Flask-powered dashboard.

The platform provides disease-specific prediction capabilities, confidence scores, dataset integration, prediction history, and an intuitive user interface.

---

## 🚀 Key Features

- **Symptom-Based Prediction:** Predict potential diseases based on selected symptoms.
- **Multiple Disease Prediction:** Supports Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer clinical prediction modules.
- **Machine Learning Integration:** Uses machine learning pipelines for clinical risk prediction.
- **Confidence Scores:** Displays prediction confidence and probability information.
- **Interactive Dashboard:** User-friendly web interface for entering health information and viewing results.
- **Dataset Integration:** Supports public datasets and custom CSV dataset uploads.
- **Prediction History:** Stores previous prediction records using SQLite.
- **Model Management:** Supports model persistence and saved model information.
- **REST API Integration:** Provides API endpoints for prediction and dataset management.

---

## 🧠 Supported Disease Modules

| Disease | Prediction Type |
|---|---|
| Diabetes | Clinical Risk Prediction |
| Heart Disease | Clinical Risk Prediction |
| Parkinson's Disease | Clinical Risk Prediction |
| Breast Cancer | Clinical Risk Prediction |
| Multiple Other Conditions | Symptom-Based Prediction |

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | HTML, CSS, JavaScript |
| Backend | Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Model Persistence | Joblib |
| Database | SQLite |
| Additional ML Library | XGBoost |

---

## 📂 Project Structure

```text
MediPredict-AI/
│
├── app.py
├── run.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── disease_predictor/
│   ├── __init__.py
│   ├── app.py
│   ├── data.py
│   ├── database.py
│   └── model.py
│
├── templates/
│   └── index.html
│
├── tests/
│   └── test_predictor.py
│
├── Training.csv
├── Testing.csv
│
└── medi_predict.db
```

---

## ⚙️ Installation & Setup

Follow these steps to run MediPredict AI locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/MediPredict-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd MediPredict-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open in Browser

```text
http://127.0.0.1:5000/
```

---

## 🔌 API Endpoints

MediPredict AI provides the following API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main application dashboard |
| GET | `/health` | Application health check |
| GET | `/api/datasets` | Dataset information |
| GET | `/api/model-status` | Saved model information |
| GET | `/api/overview` | Dashboard statistics |
| GET | `/api/history` | Retrieve prediction history |
| DELETE | `/api/history` | Clear prediction history |
| POST | `/api/predict` | Symptom-based disease prediction |
| POST | `/api/clinical-predict` | Clinical disease risk prediction |
| POST | `/api/upload-dataset` | Upload custom CSV datasets |

---

## 📊 Machine Learning Workflow

The application follows a structured machine learning workflow:

1. **Data Collection:** Collect symptom information and clinical datasets.
2. **Data Preprocessing:** Prepare and validate input data.
3. **Feature Processing:** Process symptoms and clinical health features.
4. **Model Training:** Train clinical prediction models using available datasets.
5. **Prediction:** Generate disease risk estimates based on user inputs.
6. **Result Visualization:** Display prediction results and confidence information.
7. **History Management:** Store prediction records for future reference.

---

## 🎯 Project Objectives

- Apply machine learning techniques to healthcare-related prediction problems.
- Develop an interactive disease risk prediction platform.
- Integrate multiple disease-specific clinical prediction modules.
- Understand practical implementation of machine learning with Flask.
- Demonstrate data processing, model integration, and database management.

---

## 🔮 Future Enhancements

- Integration of additional disease prediction models.
- Improved model evaluation and performance comparison.
- Advanced interactive analytics dashboard.
- User authentication and personalized health history.
- Cloud deployment and scalable API integration.
- Explainable AI for better prediction transparency.

---

## ⚠️ Disclaimer

MediPredict AI is an educational and research-oriented machine learning project. Its predictions are estimates based on available datasets and user-provided information.

**This application is NOT a medical diagnosis system and must not be used as a substitute for professional medical advice, diagnosis, or treatment.** Always consult a qualified healthcare professional for medical concerns.

---

## 👨‍💻 Developer

**Armaan Singh**


## ⭐ Support

If you find this project interesting or useful, consider giving this repository a ⭐ on GitHub!

**Built with ❤️ using Python, Machine Learning, and Flask.**
