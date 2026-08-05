"""Postgres connection and schema for the ETL pipeline.

Two tables, matching a standard ETL staging/final split:
  - raw_weather_observations: landing table for every API pull, one row per
    (source, location, timestamp) as fetched — kept close to the source
    shape for auditability/debugging.
  - turbine_readings: the transformed, unified table (same schema as
    data/raw/turbine_data.csv) that the rest of the pipeline consumes.
"""
from sqlalchemy import create_engine, text

import config

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(config.DATABASE_URL)
    return _engine


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_weather_observations (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    observed_at TIMESTAMP NOT NULL,
    wind_speed_ms DOUBLE PRECISION,
    wind_direction_deg DOUBLE PRECISION,
    temperature_c DOUBLE PRECISION,
    fetched_at TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (source, latitude, longitude, observed_at)
);

CREATE TABLE IF NOT EXISTS turbine_readings (
    id SERIAL PRIMARY KEY,
    "Time" TIMESTAMP NOT NULL,
    "WTG" TEXT NOT NULL,
    "ActivePower" DOUBLE PRECISION,
    "AmbientTemperature" DOUBLE PRECISION,
    "BearingShaftTemperature" DOUBLE PRECISION,
    "Blade1PitchAngle" DOUBLE PRECISION,
    "Blade2PitchAngle" DOUBLE PRECISION,
    "Blade3PitchAngle" DOUBLE PRECISION,
    "ControlBoxTemperature" DOUBLE PRECISION,
    "GearboxBearingTemperature" DOUBLE PRECISION,
    "GearboxOilTemperature" DOUBLE PRECISION,
    "GeneratorRPM" DOUBLE PRECISION,
    "GeneratorWinding1Temperature" DOUBLE PRECISION,
    "GeneratorWinding2Temperature" DOUBLE PRECISION,
    "HubTemperature" DOUBLE PRECISION,
    "MainBoxTemperature" DOUBLE PRECISION,
    "NacellePosition" DOUBLE PRECISION,
    "ReactivePower" DOUBLE PRECISION,
    "RotorRPM" DOUBLE PRECISION,
    "TurbineStatus" DOUBLE PRECISION,
    "WindDirection" DOUBLE PRECISION,
    "WindSpeed" DOUBLE PRECISION,
    source TEXT NOT NULL,
    loaded_at TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE ("Time", "WTG")
);
"""


def init_schema():
    with get_engine().begin() as conn:
        conn.execute(text(SCHEMA_SQL))
