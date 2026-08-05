"""
Prometheus metrics for the wind power prediction API.

Two kinds of signal are exposed on GET /metrics:

1. Live serving metrics (updated on every request): request counts/latency
   per endpoint, and a histogram/gauge of predicted power output — these
   change in real time as the React frontend (or any client) calls
   /predict, so Grafana shows genuine live traffic.

2. Pipeline metrics (re-read from disk on every scrape): the model's
   held-out performance (reports/model_comparison.csv) and the latest
   Evidently AI drift report (reports/evidently/drift_summary.json)
   produced by the `monitor` DVC stage (src/monitor.py). Re-reading on
   each scrape means Grafana reflects the latest `dvc repro` run without
   restarting the backend.
"""
import csv
import json
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram

BASE_DIR = Path(__file__).resolve().parent.parent
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.csv"
DRIFT_SUMMARY_PATH = BASE_DIR / "reports" / "evidently" / "drift_summary.json"

# --- Live serving metrics ---------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "path"]
)
PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total", "Total prediction requests served"
)
PREDICTED_POWER_KW = Histogram(
    "predicted_power_kw",
    "Distribution of predicted active power output (kW)",
    buckets=(0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000),
)
LAST_PREDICTED_POWER_KW = Gauge(
    "last_predicted_power_kw", "Most recent predicted active power output (kW)"
)

# --- Pipeline / model-quality metrics -----------------------------------

MODEL_METRIC = Gauge(
    "model_metric", "Held-out performance of the currently served model", ["metric", "model_name"]
)
DATA_DRIFT_SHARE = Gauge(
    "data_drift_share", "Fraction of monitored columns flagged as drifted (Evidently AI)"
)
DATASET_DRIFT_DETECTED = Gauge(
    "dataset_drift_detected", "1 if dataset-level drift was detected, else 0"
)
COLUMN_DRIFT_SCORE = Gauge(
    "column_drift_score", "Per-column drift score from the latest Evidently AI report", ["column"]
)


def refresh_pipeline_metrics(model_name: str) -> None:
    """Re-read reports/ on disk and update the pipeline-metric gauges."""
    if COMPARISON_PATH.exists():
        with open(COMPARISON_PATH, newline="") as f:
            for row in csv.DictReader(f):
                if row[""] == model_name:
                    for metric in ("MAE", "MSE", "RMSE", "R2"):
                        MODEL_METRIC.labels(metric=metric, model_name=model_name).set(float(row[metric]))
                    break

    if DRIFT_SUMMARY_PATH.exists():
        summary = json.loads(DRIFT_SUMMARY_PATH.read_text())
        DATA_DRIFT_SHARE.set(summary.get("drift_share", 0.0))
        DATASET_DRIFT_DETECTED.set(1 if summary.get("dataset_drift_detected") else 0)
        for column, score in summary.get("per_column_drift_score", {}).items():
            COLUMN_DRIFT_SCORE.labels(column=column).set(score)


def record_prediction(value_kw: float) -> None:
    PREDICTION_REQUESTS_TOTAL.inc()
    PREDICTED_POWER_KW.observe(value_kw)
    LAST_PREDICTED_POWER_KW.set(value_kw)
