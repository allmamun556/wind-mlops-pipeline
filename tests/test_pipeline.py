"""Unit tests exercised by the CI/CD pipeline (GitHub Actions `test` job)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402
import data_generation  # noqa: E402
import preprocessing  # noqa: E402


def test_generate_shape_and_columns():
    df = data_generation.generate(n_rows=500, seed=1)
    assert len(df) == 500
    expected_cols = {
        "Time", "WTG", "ActivePower", "AmbientTemperature", "WindSpeed", "TurbineStatus",
    }
    assert expected_cols.issubset(set(df.columns))


def test_generate_active_power_non_negative_after_dropna():
    df = data_generation.generate(n_rows=1000, seed=2)
    valid = df["ActivePower"].dropna()
    assert (valid >= 0).all()


def test_power_curve_monotonic_up_to_rated():
    speeds = np.array([0, 3, 6, 9, 12])
    power = data_generation.power_curve(speeds, rated_power=2000, cut_in=3, rated_speed=12, cut_out=25)
    assert list(power) == sorted(power)


def test_impute_removes_all_nans(tmp_path):
    df = data_generation.generate(n_rows=800, seed=3)
    imputed = preprocessing.impute(df)
    assert imputed.isna().sum().sum() == 0


def test_correlated_columns_defined_in_config():
    assert len(config.DROP_HIGH_CORRELATION) == 5
    assert "GeneratorWinding2Temperature" in config.DROP_HIGH_CORRELATION


def test_drift_summary_schema_if_present():
    path = "reports/evidently/drift_summary.json"
    if not os.path.exists(path):
        return  # monitoring stage hasn't run yet in this environment
    with open(path) as f:
        summary = json.load(f)
    for key in ["n_columns", "n_drifted_columns", "drift_share", "dataset_drift_detected"]:
        assert key in summary
