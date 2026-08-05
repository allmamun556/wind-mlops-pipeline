"""Unit tests for the ETL transform logic (pure function, no network/DB —
extract.py hits the real Open-Meteo API and load.py needs Postgres, so
those are exercised manually via `python src/etl/run_etl.py`, not in CI)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from etl.transform import derive_turbine_readings  # noqa: E402


def _fake_weather_df(n=100):
    return pd.DataFrame({
        "observed_at": pd.date_range("2026-01-01", periods=n, freq="h"),
        "wind_speed_ms": np.clip(np.random.default_rng(0).normal(8, 3, n), 0, 25),
        "wind_direction_deg": np.random.default_rng(1).uniform(0, 360, n),
        "temperature_c": np.random.default_rng(2).normal(10, 5, n),
        "source": "open-meteo-historical",
    })


def test_derive_turbine_readings_matches_synthetic_schema():
    import data_generation
    synthetic_cols = set(data_generation.generate(n_rows=10, seed=0).columns)

    weather = _fake_weather_df()
    turbine = derive_turbine_readings(weather)

    assert synthetic_cols == set(turbine.columns) - {"source"}


def test_derive_turbine_readings_preserves_real_weather_values():
    weather = _fake_weather_df()
    turbine = derive_turbine_readings(weather)

    assert (turbine["WindSpeed"].to_numpy() == weather["wind_speed_ms"].to_numpy()).all()
    assert (turbine["AmbientTemperature"].to_numpy() == weather["temperature_c"].to_numpy()).all()


def test_derive_turbine_readings_active_power_non_negative():
    weather = _fake_weather_df(n=500)
    turbine = derive_turbine_readings(weather)
    assert (turbine["ActivePower"] >= 0).all()


def test_derive_turbine_readings_row_count_matches_input():
    weather = _fake_weather_df(n=250)
    turbine = derive_turbine_readings(weather)
    assert len(turbine) == 250
