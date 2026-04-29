# api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routers import predict, health
from api.services.model_service import get_model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — load model once
    logger.info("Starting up — loading model...")
    get_model_service().load(model_path="model.pkl")
    logger.info("Model ready. API is live.")
    yield
    # Shutdown
    logger.info("Shutting down.")


app = FastAPI(
    title="Placement Risk Intelligence API",
    description="Predicts student placement probability with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(health.router)


@app.get("/")
async def root():
    return {
        "name":    "Placement Risk Intelligence API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }