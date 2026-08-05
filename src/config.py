"""Shared configuration for the wind power MLOps pipeline."""
import os

RAW_DATA_PATH = "data/raw/turbine_data.csv"
PROCESSED_TRAIN_PATH = "data/processed/train.csv"
PROCESSED_TEST_PATH = "data/processed/test.csv"
REFERENCE_DATA_PATH = "data/monitoring/reference.csv"
CURRENT_DATA_PATH = "data/monitoring/current.csv"

TARGET = "ActivePower"

# Columns dropped after correlation analysis (thesis section 5.2.2):
# each is >0.9 correlated with a retained sibling column and is dropped to
# reduce redundancy / overfitting risk.
DROP_HIGH_CORRELATION = [
    "GeneratorWinding2Temperature",  # ~0.9999 corr with GeneratorWinding1Temperature
    "GearboxBearingTemperature",     # ~0.90 corr with GearboxOilTemperature
    "RotorRPM",                      # ~0.9997 corr with GeneratorRPM
    "Blade2PitchAngle",              # highly corr with Blade1PitchAngle
    "Blade3PitchAngle",              # highly corr with Blade1PitchAngle
]

# Non-numeric / identifier columns not used as model features
NON_FEATURE_COLUMNS = ["Time", "WTG"]

IMPUTATION_STRATEGY = {
    "ActivePower": "interpolate",
    "ReactivePower": "interpolate",
    "WindSpeed": "median",
    "AmbientTemperature": "median",
    "BearingShaftTemperature": "median",
    "Blade1PitchAngle": "median",
    "ControlBoxTemperature": "median",
    "GearboxOilTemperature": "median",
    "GeneratorRPM": "median",
    "GeneratorWinding1Temperature": "median",
    "HubTemperature": "median",
    "MainBoxTemperature": "median",
    "NacellePosition": "median",
    "TurbineStatus": "median",
    "WindDirection": "median",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = "wind-power-prediction"
MODEL_REGISTRY_NAME = "wind_power_predictor"

BEST_MODEL_PATH = "models/best_model.pkl"
FEATURE_LIST_PATH = "models/feature_list.json"
