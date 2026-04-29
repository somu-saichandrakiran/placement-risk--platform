# api/routers/health.py
from fastapi import APIRouter
from api.services.model_service import get_model_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    svc = get_model_service()
    return {
        "status":        "ok",
        "model_version": svc.version,
        "model_loaded":  svc._loaded,
    }


@router.get("/ready")
async def ready():
    svc = get_model_service()
    if not svc._loaded:
        return {"status": "not ready"}
    return {"status": "ready"}