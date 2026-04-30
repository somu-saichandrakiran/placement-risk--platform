# Placement Risk Intelligence Platform

> ML platform predicting student placement probability with SHAP explanations, 
> salary estimates, and actionable recommendations.

## Architecture
Student Input → FastAPI → XGBoost (Calibrated) → SHAP Explainer
↓
Probability + CI + Salary Range + Top Drivers + Recommendations
↓
Streamlit UI (Gauge + Waterfall + What-If Simulator)

## ML Model
- **Algorithm**: XGBoost + Isotonic Calibration
- **ROC-AUC**: 0.865 | **Brier Score**: 0.08 | **CV Folds**: 5
- **Explainability**: SHAP TreeExplainer (per-prediction waterfall)
- **Tracked**: MLflow experiment registry (4 runs, model v4)

## Features
- Calibrated placement probability with 95% confidence interval
- Salary P10/P50/P90 quantile estimates
- Per-prediction SHAP waterfall explanations
- Ranked counterfactual recommendations ("do X to gain +4.5%")
- What-if simulator — adjust inputs, recalculate live
- Bias disclaimer on every prediction
- 18 automated API tests (pytest)

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate data + train
python generate_data.py
python -m ml.train
python -m ml.save_model

# 3. Start API (terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start UI (terminal 2)
streamlit run app.py

# 5. Run tests
pytest tests/test_api.py -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/predict` | Placement probability + SHAP + salary |
| POST | `/v1/counterfactual` | Ranked improvement recommendations |
| GET | `/health` | Model version + status |
| GET | `/docs` | Interactive Swagger UI |

## Project Structure
placement-risk-platform/
├── api/
│   ├── routers/        # predict, health endpoints
│   ├── schemas/        # Pydantic input/output models
│   └── services/       # ModelService (singleton, SHAP)
├── ml/
│   ├── features.py     # Feature engineering (leakage-free)
│   ├── pipeline.py     # XGBoost + calibration pipeline
│   ├── train.py        # MLflow-tracked training
│   └── save_model.py   # Serialize final model
├── tests/
│   └── test_api.py     # 18 API + model behaviour tests
├── app.py              # Streamlit frontend
├── generate_data.py    # Synthetic data generator
└── model.pkl           # Trained model artifact

## Key Engineering Decisions

**Why XGBoost over neural networks?**  
Tabular data under 10K rows — tree ensembles consistently outperform.
Native SHAP TreeExplainer support gives exact Shapley values in milliseconds.

**Why isotonic calibration?**  
Raw XGBoost probabilities cluster near 0 and 1. Isotonic regression maps
scores to proper probabilities verified by Brier score (0.08).

**Why sklearn Pipeline?**  
Prevents data leakage — StandardScaler fits only on X_train.
All transforms applied consistently at inference time.

**Why SHAP over feature importance?**  
Feature importance is biased toward high-cardinality features.
SHAP gives per-prediction, additive, theoretically grounded explanations.