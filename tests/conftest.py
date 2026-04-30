# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.services.model_service import get_model_service


@pytest.fixture(scope="session", autouse=True)
def load_model():
    """Load model once before all tests."""
    get_model_service().load(model_path="model.pkl")


@pytest.fixture(scope="session")
def client(load_model):
    with TestClient(app) as c:
        yield c