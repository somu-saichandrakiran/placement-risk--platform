# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID = {
    "tenth_percent":       78.5,
    "twelfth_percent":     72.0,
    "cgpa":                67.5,
    "employability_score": 74.0,
    "mba_percent":         62.0,
    "internship_count":    1,
    "project_count":       3,
    "hackathon_count":     1,
    "active_backlogs":     0,
    "work_exp_flag":       0,
    "specialization_12th": "Science",
    "degree_type":         "Sci&Tech",
    "specialization":      "Mkt&Fin",
    "college_tier":        "T2"
}


def test_predict_status_200():
    r = client.post("/v1/predict", json=VALID)
    assert r.status_code == 200


def test_predict_probability_range():
    r = client.post("/v1/predict", json=VALID)
    prob = r.json()["placement"]["probability"]
    assert 0.0 <= prob <= 1.0


def test_confidence_interval_valid():
    r = client.post("/v1/predict", json=VALID)
    ci = r.json()["placement"]["confidence_interval"]
    assert len(ci) == 2
    assert ci[0] <= ci[1]


def test_predict_has_drivers():
    r = client.post("/v1/predict", json=VALID)
    assert len(r.json()["top_drivers"]) >= 1


def test_predict_has_recommendations():
    r = client.post("/v1/predict", json=VALID)
    assert "top_recommendations" in r.json()


def test_predict_has_salary():
    r = client.post("/v1/predict", json=VALID)
    salary = r.json()["salary"]
    assert "p50_inr" in salary


def test_predict_has_disclaimer():
    r = client.post("/v1/predict", json=VALID)
    assert r.json()["bias_disclaimer"]


def test_predict_has_model_version():
    r = client.post("/v1/predict", json=VALID)
    assert r.json()["model_version"]


def test_invalid_cgpa_rejected():
    bad = {**VALID, "cgpa": 150.0}
    r = client.post("/v1/predict", json=bad)
    assert r.status_code == 422


def test_negative_backlogs_rejected():
    bad = {**VALID, "active_backlogs": -1}
    r = client.post("/v1/predict", json=bad)
    assert r.status_code == 422


def test_invalid_tenth_percent_rejected():
    bad = {**VALID, "tenth_percent": 110.0}
    r = client.post("/v1/predict", json=bad)
    assert r.status_code == 422


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] == True


def test_ready_endpoint():
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_root_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


def test_higher_cgpa_increases_probability():
    low  = {**VALID, "cgpa": 45.0}
    high = {**VALID, "cgpa": 90.0}
    prob_low  = client.post("/v1/predict", json=low).json()["placement"]["probability"]
    prob_high = client.post("/v1/predict", json=high).json()["placement"]["probability"]
    assert prob_high > prob_low


def test_backlogs_reduce_probability(client):
    clean = {**VALID, "cgpa": 55.0, "active_backlogs": 0}
    dirty = {**VALID, "cgpa": 55.0, "active_backlogs": 5}
    prob_clean = client.post("/v1/predict", json=clean).json()["placement"]["probability"]
    prob_dirty = client.post("/v1/predict", json=dirty).json()["placement"]["probability"]
    assert prob_clean >= prob_dirty

def test_counterfactual_endpoint():
    r = client.post("/v1/counterfactual", json=VALID)
    assert r.status_code == 200
    data = r.json()
    assert "base_probability" in data
    assert "recommendations" in data


def test_work_experience_helps():
    no_exp  = {**VALID, "work_exp_flag": 0}
    has_exp = {**VALID, "work_exp_flag": 1}
    prob_no  = client.post("/v1/predict", json=no_exp).json()["placement"]["probability"]
    prob_yes = client.post("/v1/predict", json=has_exp).json()["placement"]["probability"]
    assert prob_yes >= prob_no