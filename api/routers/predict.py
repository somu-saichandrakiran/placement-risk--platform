# api/routers/predict.py
import time
from fastapi import APIRouter, Depends
from api.schemas.student import StudentInput
from api.schemas.prediction import (
    PredictionResponse, PlacementResult,
    SalaryResult, DriverResult, RecommendationResult
)
from api.services.model_service import get_model_service, ModelService

router = APIRouter(prefix="/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    student: StudentInput,
    model_svc: ModelService = Depends(get_model_service),
):
    start = time.monotonic()
    result = model_svc.predict(student.model_dump())
    latency = round((time.monotonic() - start) * 1000, 1)

    return PredictionResponse(
        model_version=result["model_version"],
        placement=PlacementResult(**result["placement"]),
        salary=SalaryResult(**result["salary"]),
        top_drivers=[
            DriverResult(**d) for d in result["top_drivers"]
        ],
        top_recommendations=[
            RecommendationResult(**r) for r in result["top_recommendations"]
        ],
        latency_ms=latency,
        bias_disclaimer=result["bias_disclaimer"],
    )


@router.post("/counterfactual")
async def counterfactual(
    student: StudentInput,
    model_svc: ModelService = Depends(get_model_service),
):
    result = model_svc.predict(student.model_dump())
    return {
        "base_probability":  result["placement"]["probability"],
        "recommendations":   result["top_recommendations"],
    }