# api/schemas/prediction.py
from pydantic import BaseModel
from typing import Optional

class PlacementResult(BaseModel):
    probability:         float
    confidence_interval: list[float]
    calibrated:          bool

class SalaryResult(BaseModel):
    p10_inr: Optional[int]
    p50_inr: Optional[int]
    p90_inr: Optional[int]
    note:    Optional[str] = None

class DriverResult(BaseModel):
    feature:   str
    shap_value: float
    impact_pct: float
    direction:  str
    message:    str

class RecommendationResult(BaseModel):
    action:            str
    current_value:     float
    new_value:         float
    probability_gain:  float
    message:           str

class PredictionResponse(BaseModel):
    model_version:       str
    placement:           PlacementResult
    salary:              SalaryResult
    top_drivers:         list[DriverResult]
    top_recommendations: list[RecommendationResult]
    latency_ms:          float
    bias_disclaimer:     str