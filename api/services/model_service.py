# api/services/model_service.py
import pickle
import numpy as np
import pandas as pd
import shap
from threading import Lock
from loguru import logger
from ml.features import load_and_clean
from ml.pipeline import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
)

ACTIONABLE_FEATURES = {
    "internship_count": {"label": "Complete an internship",   "step": 1,  "min": 0, "max": 10},
    "project_count":    {"label": "Add a project",            "step": 1,  "min": 0, "max": 20},
    "hackathon_count":  {"label": "Join a hackathon",         "step": 1,  "min": 0, "max": 10},
    "active_backlogs":  {"label": "Clear a backlog",          "step": -1, "min": 0, "max": 20},
    "work_exp_flag":    {"label": "Gain work experience",     "step": 1,  "min": 0, "max": 1},
}

DISCLAIMER = (
    "This prediction reflects historical placement patterns "
    "and may contain bias. Use as one signal, not a final verdict."
)


class ModelService:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
        return cls._instance

    def load(self, model_path: str = "model.pkl"):
        if self._loaded:
            return
        logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.version = "v4-local"

        # Build SHAP explainer from inner XGBoost model
        calibrated   = self.pipeline.calibrated_classifiers_[0].estimator
        preprocessor = calibrated.named_steps["preprocessor"]
        classifier   = calibrated.named_steps["classifier"]

        # Get feature names after preprocessing
        self.feature_names = (
            NUMERIC_FEATURES +
            CATEGORICAL_FEATURES +
            BINARY_FEATURES
        )

        self.explainer   = shap.TreeExplainer(classifier)
        self.preprocessor_inner = preprocessor
        self.engineer    = calibrated.named_steps["engineer"]
        self._loaded     = True
        logger.info(f"Model loaded — version {self.version}")

        # Smoke test
        self._smoke_test()

    def _smoke_test(self):
        test = {
            "tenth_percent": 75.0, "twelfth_percent": 72.0,
            "cgpa": 67.0, "employability_score": 74.0,
            "mba_percent": 62.0, "internship_count": 1,
            "project_count": 2, "hackathon_count": 0,
            "active_backlogs": 0, "work_exp_flag": 0,
            "specialization_12th": "Science",
            "degree_type": "Sci&Tech",
            "specialization": "Mkt&Fin",
            "college_tier": "T2",
        }
        result = self.predict(test)
        prob = result["placement"]["probability"]
        assert 0.0 <= prob <= 1.0, f"Smoke test failed: {prob}"
        logger.info(f"Smoke test passed — prob={prob:.3f}")

    def _confidence_interval(self, prob: float, n: int = 150):
        z = 1.96
        centre = (prob + z**2 / (2*n)) / (1 + z**2/n)
        margin  = z * np.sqrt(
            (prob*(1-prob)/n + z**2/(4*n**2))
        ) / (1 + z**2/n)
        return [
            round(max(0.0, centre - margin), 3),
            round(min(1.0, centre + margin), 3),
        ]

    def predict(self, raw: dict) -> dict:
        X = pd.DataFrame([raw])
        prob = float(self.pipeline.predict_proba(X)[0][1])
        ci   = self._confidence_interval(prob)

        salary = self._predict_salary(raw, prob)
        drivers, recommendations = self._explain(raw, prob)

        return {
            "model_version": self.version,
            "placement": {
                "probability":         round(prob, 3),
                "confidence_interval": ci,
                "calibrated":          True,
            },
            "salary": salary,
            "top_drivers":         drivers[:3],
            "top_recommendations": recommendations[:3],
            "bias_disclaimer":     DISCLAIMER,
        }

    def _predict_salary(self, raw: dict, prob: float) -> dict:
        if prob < 0.3:
            return {
                "p10_inr": None, "p50_inr": None, "p90_inr": None,
                "note": "Salary estimate requires higher placement probability",
            }
        base   = 250000
        cgpa   = raw.get("cgpa", 60)
        etest  = raw.get("employability_score", 60)
        workex = raw.get("work_exp_flag", 0)

        p50 = int(base + cgpa*3000 + etest*2000 + workex*80000)
        p10 = int(p50 * 0.70)
        p90 = int(p50 * 1.45)

        return {"p10_inr": p10, "p50_inr": p50, "p90_inr": p90, "note": None}

    def _explain(self, raw: dict, base_prob: float):
        X    = pd.DataFrame([raw])
        X_eng = self.engineer.transform(X)
        X_pre = self.preprocessor_inner.transform(X_eng)

        shap_vals = self.explainer.shap_values(X_pre)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = shap_vals[0]

        all_features = (
            NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
        )
        drivers = []
        for feat, val in zip(all_features, shap_vals):
            direction = "boosts" if val > 0 else "reduces"
            drivers.append({
                "feature":    feat,
                "shap_value": round(float(val), 4),
                "impact_pct": round(abs(float(val)) * 100, 1),
                "direction":  direction,
                "message": (
                    f"{feat.replace('_',' ').title()} "
                    f"{direction} your placement by "
                    f"{abs(float(val)):.1%}"
                ),
            })
        drivers = sorted(drivers, key=lambda x: abs(x["shap_value"]), reverse=True)

        # Counterfactuals
        recommendations = []
        for feat, cfg in ACTIONABLE_FEATURES.items():
            if feat not in raw:
                continue
            modified = raw.copy()
            new_val  = raw[feat] + cfg["step"]
            new_val  = max(cfg["min"], min(cfg["max"], new_val))
            modified[feat] = new_val

            new_prob = float(
                self.pipeline.predict_proba(pd.DataFrame([modified]))[0][1]
            )
            delta = new_prob - base_prob
            if abs(delta) > 0.002:
                recommendations.append({
                    "action":           cfg["label"],
                    "current_value":    float(raw[feat]),
                    "new_value":        float(new_val),
                    "probability_gain": round(delta, 3),
                    "message": (
                        f"{'Gain' if delta > 0 else 'Note'}: "
                        f"{abs(delta):.1%} "
                        f"{'improvement' if delta > 0 else 'change'}"
                    ),
                })
        recommendations = sorted(
            recommendations,
            key=lambda x: x["probability_gain"],
            reverse=True,
        )
        return drivers, recommendations


# Singleton accessor
_service = ModelService()

def get_model_service() -> ModelService:
    return _service