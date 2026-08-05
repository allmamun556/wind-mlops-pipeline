"""
FastAPI serving layer for the wind power prediction model.

Loads the best model selected during MLflow-tracked training
(models/best_model.pkl, produced by src/train.py) and exposes it over
HTTP for the React frontend (frontend/) or any other client.

Run locally:
    uvicorn backend.main:app --reload --port 8000

In production this is what backend/Dockerfile builds and what
docker-compose.yml wires up behind the frontend's Nginx reverse proxy.
"""
import csv
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from backend import metrics

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"

# Same feature ranges/labels as app.py (Streamlit UI) — kept in sync by hand
# since both are thin serving layers over the same model bundle.
FEATURE_META = {
    "WindSpeed": {"label": "Wind Speed (m/s)", "min": 0.0, "max": 25.0, "default": 8.0, "unit": "m/s"},
    "GeneratorRPM": {"label": "Generator RPM", "min": 800.0, "max": 1800.0, "default": 1050.0, "unit": "RPM"},
    "GearboxOilTemperature": {"label": "Gearbox Oil Temperature", "min": 10.0, "max": 90.0, "default": 45.0, "unit": "°C"},
    "BearingShaftTemperature": {"label": "Bearing Shaft Temperature", "min": -10.0, "max": 90.0, "default": 30.0, "unit": "°C"},
    "GeneratorWinding1Temperature": {"label": "Generator Winding 1 Temperature", "min": -10.0, "max": 130.0, "default": 60.0, "unit": "°C"},
    "HubTemperature": {"label": "Hub Temperature", "min": -10.0, "max": 60.0, "default": 20.0, "unit": "°C"},
    "ReactivePower": {"label": "Reactive Power", "min": -300.0, "max": 400.0, "default": 60.0, "unit": "kVAR"},
    "Blade1PitchAngle": {"label": "Blade 1 Pitch Angle", "min": -5.0, "max": 35.0, "default": 2.0, "unit": "deg"},
    "AmbientTemperature": {"label": "Ambient Temperature", "min": -15.0, "max": 35.0, "default": 12.0, "unit": "°C"},
    "ControlBoxTemperature": {"label": "Control Box Temperature", "min": -10.0, "max": 60.0, "default": 25.0, "unit": "°C"},
    "MainBoxTemperature": {"label": "Main Box Temperature", "min": -10.0, "max": 60.0, "default": 22.0, "unit": "°C"},
    "NacellePosition": {"label": "Nacelle Position", "min": 0.0, "max": 360.0, "default": 180.0, "unit": "deg"},
    "WindDirection": {"label": "Wind Direction", "min": 0.0, "max": 360.0, "default": 180.0, "unit": "deg"},
    "TurbineStatus": {"label": "Turbine Status (0=stopped, 1=normal, 2=cut-out)", "min": 0.0, "max": 2.0, "default": 1.0, "unit": ""},
}

app = FastAPI(
    title="Wind Power Prediction API",
    description="Serves the MLflow-tracked wind power regression model trained in src/train.py.",
    version="1.0.0",
)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.url.path
    if path != "/metrics":
        metrics.HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, path=path).observe(duration)
        metrics.HTTP_REQUESTS_TOTAL.labels(method=request.method, path=path, status=response.status_code).inc()
    return response


_model = None
_features: list[str] = []
_model_name: Optional[str] = None


def get_model():
    global _model, _features, _model_name
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"No trained model found at {MODEL_PATH} — run `dvc repro` or `python src/train.py` first.",
            )
        bundle = joblib.load(MODEL_PATH)
        _model, _features, _model_name = bundle["model"], bundle["features"], bundle["name"]
    return _model, _features, _model_name


def get_metrics(model_name: str) -> dict:
    if not COMPARISON_PATH.exists():
        return {}
    with open(COMPARISON_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row[""] == model_name:
                return {
                    "MAE": float(row["MAE"]),
                    "MSE": float(row["MSE"]),
                    "RMSE": float(row["RMSE"]),
                    "R2": float(row["R2"]),
                }
    return {}


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature name -> value, matching /model-info's feature list")


class PredictionResponse(BaseModel):
    prediction: float
    unit: str = "kW"
    model_name: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics_endpoint():
    try:
        _, _, model_name = get_model()
        metrics.refresh_pipeline_metrics(model_name)
    except HTTPException:
        pass  # no model yet — still serve live serving metrics
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info():
    _, features, model_name = get_model()
    return {
        "model_name": model_name,
        "features": features,
        "feature_meta": {f: FEATURE_META.get(f, {"label": f, "min": None, "max": None, "default": 0.0, "unit": ""}) for f in features},
        "metrics": get_metrics(model_name),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model, features, model_name = get_model()

    missing = [f for f in features if f not in request.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")

    input_df = pd.DataFrame([{f: request.features[f] for f in features}])[features]
    raw_prediction = model.predict(input_df)[0]
    prediction = max(0.0, float(raw_prediction))
    metrics.record_prediction(prediction)

    return PredictionResponse(prediction=prediction, model_name=model_name)
