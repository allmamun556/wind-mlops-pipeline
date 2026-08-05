"""Transform: merge real weather observations into the unified turbine
schema (data/raw/turbine_data.csv format), deriving turbine-internal sensor
readings from the real wind speed / temperature.

No public API exposes real turbine-internal telemetry (bearing/generator/
gearbox temperatures, RPM, pitch angles, etc.) — that data is proprietary
to operators. The physics-based derivation below is the same one
src/data_generation.py uses, just driven by real wind speed/temperature
instead of synthetic values, so ActivePower and the thermal/drivetrain
features stay internally consistent with each other.
"""
import numpy as np
import pandas as pd

from data_generation import power_curve


def derive_turbine_readings(weather_df: pd.DataFrame, seed: int = 42, n_turbines: int = 5) -> pd.DataFrame:
    """weather_df: columns [observed_at, wind_speed_ms, wind_direction_deg, temperature_c]."""
    rng = np.random.default_rng(seed)
    n = len(weather_df)

    wind_speed = weather_df["wind_speed_ms"].to_numpy(dtype=float)
    wind_direction = weather_df["wind_direction_deg"].to_numpy(dtype=float)
    ambient_temp = weather_df["temperature_c"].to_numpy(dtype=float)

    active_power = power_curve(wind_speed) + rng.normal(0, 25, n)
    active_power = np.clip(active_power, 0, None)
    reactive_power = 0.3 * active_power + rng.normal(0, 15, n)

    generator_rpm = 900 + 8.2 * wind_speed + 0.01 * active_power + rng.normal(0, 15, n)
    rotor_rpm = generator_rpm / 90.0 + rng.normal(0, 0.05, n)

    nacelle_position = (wind_direction + rng.normal(0, 5, n)) % 360
    turbine_status = np.where(wind_speed < 3.0, 0, np.where(wind_speed > 25.0, 2, 1))

    blade1_pitch = np.where(wind_speed > 12, (wind_speed - 12) * 2.5, 0) + rng.normal(0, 0.4, n)
    blade2_pitch = blade1_pitch + rng.normal(0, 0.15, n)
    blade3_pitch = blade1_pitch + rng.normal(0, 0.15, n)

    load_factor = active_power / 2000.0
    bearing_shaft_temp = ambient_temp + 25 * load_factor + rng.normal(0, 2.5, n)
    gearbox_oil_temp = ambient_temp + 40 * load_factor + rng.normal(0, 3.0, n)
    gearbox_bearing_temp = gearbox_oil_temp * 0.95 + rng.normal(0, 1.5, n)
    generator_winding1_temp = ambient_temp + 55 * load_factor + rng.normal(0, 3.5, n)
    generator_winding2_temp = generator_winding1_temp + rng.normal(0, 0.3, n)
    hub_temp = ambient_temp + 10 * load_factor + rng.normal(0, 2.0, n)
    control_box_temp = ambient_temp + 15 * load_factor + rng.normal(0, 2.0, n)
    main_box_temp = ambient_temp + 12 * load_factor + rng.normal(0, 2.0, n)

    wtg = rng.integers(1, n_turbines + 1, size=n)

    return pd.DataFrame({
        "Time": weather_df["observed_at"].values,
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
        "source": weather_df["source"].values,
    })
