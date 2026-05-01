# 🎯 Placement Risk Intelligence Platform

A production-grade machine learning platform that predicts student placement probability with calibrated confidence, salary projections, and explainable AI insights.

---

## 🚀 Overview

This system goes beyond simple prediction and delivers decision intelligence:

* 📊 Calibrated placement probability with **95% confidence interval**
* 💰 Salary estimates (**P10 / P50 / P90 quantiles**)
* 🔍 Per-prediction **SHAP explanations (waterfall)**
* 📈 Counterfactual recommendations (*“do X → +4.5% improvement”*)
* 🎛️ What-if simulator for real-time analysis
* ⚖️ Bias disclaimer for ethical transparency

---

## 🏗️ Architecture

```
Student Input
     ↓
FastAPI Backend
     ↓
XGBoost (Calibrated Model)
     ↓
SHAP Explainer
     ↓
----------------------------------------
Probability + Confidence Interval
Salary Estimates (P10 / P50 / P90)
Top Feature Drivers
Actionable Recommendations
----------------------------------------
     ↓
Streamlit UI (Gauge + Waterfall + Simulator)
```

---

## 🧠 Machine Learning Model

* **Algorithm:** XGBoost + Isotonic Calibration
* **ROC-AUC:** 0.865
* **Brier Score:** 0.08
* **Cross Validation:** 5-Fold
* **Explainability:** SHAP TreeExplainer
* **Experiment Tracking:** MLflow (4 runs, model v4)

---

## ⚙️ Core Features

* Calibrated probability output (reliable, not raw scores)
* Confidence interval estimation (uncertainty-aware predictions)
* Salary distribution modeling (quantile regression)
* SHAP-based per-instance explanations
* Ranked counterfactual recommendations
* Real-time what-if simulation
* Bias disclaimer on predictions
* 18 automated API tests (pytest)

---

## ⚡ Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Generate Data & Train Model

```bash
python generate_data.py
python -m ml.train
python -m ml.save_model
```

---

### 3️⃣ Start Backend (FastAPI)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

📍 API Docs: http://127.0.0.1:8000/docs

---

### 4️⃣ Start Frontend (Streamlit)

```bash
streamlit run app.py
```

---

### 5️⃣ Run Tests

```bash
pytest tests/test_api.py -v
```

---

## 🔌 API Endpoints

| Method | Endpoint           | Description                           |
| ------ | ------------------ | ------------------------------------- |
| POST   | /v1/predict        | Placement probability + SHAP + salary |
| POST   | /v1/counterfactual | Ranked improvement recommendations    |
| GET    | /health            | Model status + version                |
| GET    | /docs              | Swagger UI                            |

---

## 📁 Project Structure

```
placement-risk-platform/
├── api/
│   ├── routers/        # API endpoints
│   ├── schemas/        # Pydantic models
│   └── services/       # Model logic + SHAP
│
├── ml/
│   ├── features.py     # Feature engineering (leakage-free)
│   ├── pipeline.py     # XGBoost + calibration pipeline
│   ├── train.py        # MLflow training pipeline
│   └── save_model.py   # Model serialization
│
├── tests/
│   └── test_api.py     # API + model tests (18 cases)
│
├── app.py              # Streamlit frontend
├── generate_data.py    # Synthetic data generator
├── model.pkl           # Trained model artifact
│
├── requirements.txt
└── README.md
```

---

## 🧩 Key Engineering Decisions

### Why XGBoost over Neural Networks?

For structured/tabular data (<10K rows), tree-based models consistently outperform deep learning.
XGBoost provides:

* Strong performance on tabular data
* Built-in handling of feature interactions
* Native compatibility with SHAP TreeExplainer

---

### Why Isotonic Calibration?

Raw XGBoost probabilities are often overconfident.
Isotonic regression maps model outputs to **true probability estimates**, validated using:

* **Brier Score = 0.08 (well-calibrated)**

---

### Why sklearn Pipeline?

Ensures:

* No data leakage
* Consistent preprocessing during training and inference
* Reproducibility

---

### Why SHAP instead of Feature Importance?

Traditional feature importance is:

* Global (not instance-specific)
* Biased toward high-cardinality features

SHAP provides:

* Local explanations per prediction
* Additive feature contributions
* Strong theoretical foundation (Shapley values)

---

## 📊 Example API Request

```json
POST /v1/predict

{
  "cgpa": 8.2,
  "internships": 2,
  "projects": 3,
  "certifications": 1,
  "communication_skills": 7
}
```

---

## 📈 Future Improvements

* Dockerized deployment
* CI/CD pipeline (GitHub Actions)
* Real-world dataset integration
* Model monitoring & drift detection

---

## 👤 Author

Kiran

