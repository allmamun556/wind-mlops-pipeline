"""
Synthetic Wind Turbine SCADA Data Generator
--------------------------------------------
Generates a physically-plausible wind turbine SCADA dataset that mirrors the
feature set, correlation structure, and missing-value patterns described in
the reference thesis ("Modern ML-CI/CD, Experiment Tracking and Monitoring
for Wind Power Prediction Data"), Chapter 5.

A public SCADA dataset (e.g. Kaggle wind turbine datasets) can be dropped
into data/raw/turbine_data.csv with matching column names and this script
skipped -- the rest of the pipeline is dataset-agnostic as long as the
column names in config.py are respected.

Usage:
    python src/data_generation.py --rows 40000 --seed 42 --out data/raw/turbine_data.csv
"""
import argparse
import numpy as np
import pandas as pd


def power_curve(wind_speed, rated_power=2000.0, cut_in=3.0, rated_speed=12.0, cut_out=25.0):
    """Idealised cubic wind turbine power curve (kW)."""
    power = np.zeros_like(wind_speed)
    ramp = (wind_speed >= cut_in) & (wind_speed < rated_speed)
    power[ramp] = rated_power * ((wind_speed[ramp] - cut_in) / (rated_speed - cut_in)) ** 3
    rated = (wind_speed >= rated_speed) & (wind_speed < cut_out)
    power[rated] = rated_power
    return power


def generate(n_rows: int, seed: int = 42, n_turbines: int = 5, drift_fraction: float = 0.15):
    rng = np.random.default_rng(seed)

    time_index = pd.date_range("2024-01-01", periods=n_rows, freq="10min")
    wtg = rng.integers(1, n_turbines + 1, size=n_rows)

    # --- Base weather / operational drivers -----------------------------
    day_of_year = time_index.dayofyear.values
    seasonal = 2.0 * np.sin(2 * np.pi * day_of_year / 365.0)
    wind_speed = np.clip(
        7.0 + seasonal + rng.normal(0, 3.2, n_rows) + 1.5 * np.sin(np.arange(n_rows) / 50.0),
        0, 30,
    )
    wind_direction = (180 + 60 * np.sin(np.arange(n_rows) / 300.0) + rng.normal(0, 25, n_rows)) % 360
    ambient_temp = 10 + 8 * np.sin(2 * np.pi * (day_of_year - 172) / 365.0) + rng.normal(0, 2.0, n_rows)

    # --- Power output (target) ------------------------------------------
    active_power = power_curve(wind_speed) + rng.normal(0, 25, n_rows)
    active_power = np.clip(active_power, 0, None)
    reactive_power = 0.3 * active_power + rng.normal(0, 15, n_rows)

    # --- Drivetrain (highly correlated with power / wind) ----------------
    generator_rpm = 900 + 8.2 * wind_speed + 0.01 * active_power + rng.normal(0, 15, n_rows)
    rotor_rpm = generator_rpm / 90.0 + rng.normal(0, 0.05, n_rows)  # near-perfect corr with GeneratorRPM

    nacelle_position = (wind_direction + rng.normal(0, 5, n_rows)) % 360
    turbine_status = np.where(wind_speed < 3.0, 0, np.where(wind_speed > 25.0, 2, 1))

    # --- Blade pitch angles (near-identical across 3 blades) -------------
    blade1_pitch = np.where(wind_speed > 12, (wind_speed - 12) * 2.5, 0) + rng.normal(0, 0.4, n_rows)
    blade2_pitch = blade1_pitch + rng.normal(0, 0.15, n_rows)
    blade3_pitch = blade1_pitch + rng.normal(0, 0.15, n_rows)

    # --- Thermal system (correlated with load / ambient) -----------------
    load_factor = active_power / 2000.0
    bearing_shaft_temp = ambient_temp + 25 * load_factor + rng.normal(0, 2.5, n_rows)
    gearbox_oil_temp = ambient_temp + 40 * load_factor + rng.normal(0, 3.0, n_rows)
    gearbox_bearing_temp = gearbox_oil_temp * 0.95 + rng.normal(0, 1.5, n_rows)  # highly corr w/ oil temp
    generator_winding1_temp = ambient_temp + 55 * load_factor + rng.normal(0, 3.5, n_rows)
    generator_winding2_temp = generator_winding1_temp + rng.normal(0, 0.3, n_rows)  # near-duplicate
    hub_temp = ambient_temp + 10 * load_factor + rng.normal(0, 2.0, n_rows)
    control_box_temp = ambient_temp + 15 * load_factor + rng.normal(0, 2.0, n_rows)
    main_box_temp = ambient_temp + 12 * load_factor + rng.normal(0, 2.0, n_rows)

    df = pd.DataFrame({
        "Time": time_index,
        "WTG": [f"WTG_{i:02d}" for i in wtg],
        "ActivePower": active_power,
        "AmbientTemperature": ambient_temp,
        "BearingShaftTemperature": bearing_shaft_temp,
        "Blade1PitchAngle": blade1_pitch,
        "Blade2PitchAngle": blade2_pitch,
        "Blade3PitchAngle": blade3_pitch,
        "ControlBoxTemperature": control_box_temp,
        "GearboxBearingTemperature": gearbox_bearing_temp,
        "GearboxOilTemperature": gearbox_oil_temp,
        "GeneratorRPM": generator_rpm,
        "GeneratorWinding1Temperature": generator_winding1_temp,
        "GeneratorWinding2Temperature": generator_winding2_temp,
        "HubTemperature": hub_temp,
        "MainBoxTemperature": main_box_temp,
        "NacellePosition": nacelle_position,
        "ReactivePower": reactive_power,
        "RotorRPM": rotor_rpm,
        "TurbineStatus": turbine_status,
        "WindDirection": wind_direction,
        "WindSpeed": wind_speed,
    })

    # --- Inject missing values at rates proportional to the thesis table -
    missing_rate = {
        "ActivePower": 0.02, "AmbientTemperature": 0.02, "BearingShaftTemperature": 0.05,
        "Blade1PitchAngle": 0.06, "Blade2PitchAngle": 0.06, "Blade3PitchAngle": 0.06,
        "ControlBoxTemperature": 0.05, "GearboxBearingTemperature": 0.05, "GearboxOilTemperature": 0.05,
        "GeneratorRPM": 0.05, "GeneratorWinding1Temperature": 0.05, "GeneratorWinding2Temperature": 0.05,
        "HubTemperature": 0.05, "MainBoxTemperature": 0.05, "NacellePosition": 0.04,
        "ReactivePower": 0.02, "RotorRPM": 0.05, "TurbineStatus": 0.05,
        "WindDirection": 0.04, "WindSpeed": 0.02,
    }
    for col, rate in missing_rate.items():
        mask = rng.random(n_rows) < rate
        df.loc[mask, col] = np.nan

    # --- Inject a drift segment in the tail of the series (for Evidently)-
    # Simulates a seasonal / sensor-calibration shift so the monitoring
    # stage has something real to detect.
    drift_start = int(n_rows * (1 - drift_fraction))
    df.loc[drift_start:, "AmbientTemperature"] += 4.0
    df.loc[drift_start:, "GearboxOilTemperature"] += 6.0
    df.loc[drift_start:, "WindSpeed"] *= 0.85
    df.loc[drift_start:, "GeneratorWinding1Temperature"] += 5.0

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic wind turbine SCADA data")
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/turbine_data.csv")
    args = parser.parse_args()

    data = generate(args.rows, args.seed)
    data.to_csv(args.out, index=False)
    print(f"Generated {len(data):,} rows x {data.shape[1]} columns -> {args.out}")
    print(data.isna().sum())
