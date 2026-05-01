# 🎯 Placement Risk Intelligence Platform

A production-oriented machine learning platform that predicts student placement probability with calibrated confidence, salary projections, and explainable AI insights.

---

## 🚀 Overview

This system goes beyond binary prediction. It delivers:

- 📊 Calibrated placement probability with 95% confidence interval  
- 💰 Salary estimates (P10 / P50 / P90 quantiles)  
- 🔍 Per-prediction SHAP explanations (waterfall)  
- 📈 Counterfactual recommendations ("do X → +4.5% improvement")  
- 🎛️ What-if simulator for real-time decision analysis  

---

## 🏗️ Architecture

Student Input
↓
FastAPI Backend
↓
XGBoost (Calibrated)
↓
SHAP Explainer
↓
Probability + Confidence Interval
Salary Estimates
Top Feature Drivers
Actionable Recommendations

 ↓
 Streamlit UI (Gauge + Waterfall + Simulator)

 
---

## 🧠 Machine Learning Model

- **Algorithm:** XGBoost + Isotonic Calibration  
- **ROC-AUC:** 0.865  
- **Brier Score:** 0.08  
- **Cross Validation:** 5-Fold  
- **Explainability:** SHAP TreeExplainer  
- **Tracking:** MLflow (4 runs, model v4)

---

## ⚙️ Core Features

- Calibrated probability output (reliable, not raw scores)
- Confidence interval estimation (uncertainty-aware predictions)
- Salary distribution modeling (P10/P50/P90)
- SHAP-based per-instance explanations
- Ranked counterfactual recommendations
- Real-time what-if simulation
- Bias disclaimer for ethical transparency
- 18 automated API tests (pytest)

---

## ⚡ Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt

2️⃣ Generate Data & Train Model

python generate_data.py
python -m ml.train
python -m ml.save_model

3️⃣ Start Backend

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

4️⃣ Start Frontend

streamlit run app.py

5️⃣ Run Tests
pytest tests/test_api.py -v
🔌 API Endpoints
Method	Endpoint	Description
POST	/v1/predict	Placement probability + SHAP + salary
POST	/v1/counterfactual	Ranked improvement recommendations
GET	/health	Model status + version
GET	/docs	Swagger UI
📁 Project Structure
placement-risk-platform/
├── api/
│   ├── routers/
│   ├── schemas/
│   └── services/
├── ml/
│   ├── features.py
│   ├── pipeline.py
│   ├── train.py
│   └── save_model.py
├── tests/
│   └── test_api.py
├── app.py
├── generate_data.py
└── model.pkl
