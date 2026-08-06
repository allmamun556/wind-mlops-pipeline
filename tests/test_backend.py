"""FastAPI backend tests — exercised by the CI/CD pipeline (`test` job)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_model_info():
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    assert body["model_name"]
    assert isinstance(body["features"], list) and len(body["features"]) > 0
    assert set(body["feature_meta"]) == set(body["features"])
    assert "R2" in body["metrics"]


def test_predict_success():
    info = client.get("/model-info").json()
    features = {f: info["feature_meta"][f]["default"] for f in info["features"]}
    res = client.post("/predict", json={"features": features})
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] >= 0.0
    assert body["unit"] == "kW"
    assert body["model_name"] == info["model_name"]


def test_predict_missing_features():
    res = client.post("/predict", json={"features": {}})
    assert res.status_code == 422
    assert "Missing required features" in res.json()["detail"]


def test_metrics_endpoint_exposes_prometheus_format():
    client.get("/model-info")  # ensure a model is loaded before scraping
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "prediction_requests_total" in res.text
