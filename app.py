"""
Streamlit Wind Power Prediction App (thesis Section 5.3.2.7)
---------------------------------------------------------------
Interactive interface for wind power output prediction from turbine SCADA
operating conditions, served by the best model selected during the MLflow
experimentation stage (src/train.py -> models/best_model.pkl).

Run locally:
    streamlit run app.py

In the full pipeline this app is what GitHub Actions pushes to the Hugging
Face Hub as a Space on every successful build (see
.github/workflows/ci-cd.yml).
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wind Power Prediction", layout="wide")

FEATURE_META = {
    "WindSpeed": ("Wind Speed (m/s)", 0.0, 25.0, 8.0),
    "GeneratorRPM": ("Generator RPM", 800.0, 1800.0, 1050.0),
    "GearboxOilTemperature": ("Gearbox Oil Temperature (°C)", 10.0, 90.0, 45.0),
    "BearingShaftTemperature": ("Bearing Shaft Temperature (°C)", -10.0, 90.0, 30.0),
    "GeneratorWinding1Temperature": ("Generator Winding 1 Temperature (°C)", -10.0, 130.0, 60.0),
    "HubTemperature": ("Hub Temperature (°C)", -10.0, 60.0, 20.0),
    "ReactivePower": ("Reactive Power (kVAR)", -300.0, 400.0, 60.0),
    "Blade1PitchAngle": ("Blade 1 Pitch Angle (deg)", -5.0, 35.0, 2.0),
    "AmbientTemperature": ("Ambient Temperature (°C)", -15.0, 35.0, 12.0),
    "ControlBoxTemperature": ("Control Box Temperature (°C)", -10.0, 60.0, 25.0),
    "MainBoxTemperature": ("Main Box Temperature (°C)", -10.0, 60.0, 22.0),
    "NacellePosition": ("Nacelle Position (deg)", 0.0, 360.0, 180.0),
    "WindDirection": ("Wind Direction (deg)", 0.0, 360.0, 180.0),
    "TurbineStatus": ("Turbine Status (0=stopped,1=normal,2=cut-out)", 0, 2, 1),
}


@st.cache_resource
def load_model():
    bundle = joblib.load("models/best_model.pkl")
    return bundle["model"], bundle["features"], bundle["name"]


def main():
    st.title("🌬️ Wind Power Prediction")
    st.caption(
        "MLOps pipeline demo — model served from the MLflow Model Registry "
        "(trained on correlation-pruned, imputed SCADA features)."
    )

    try:
        model, features, model_name = load_model()
    except FileNotFoundError:
        st.error("No trained model found at models/best_model.pkl — run `python src/train.py` first.")
        return

    st.sidebar.header("Input Features")
    st.sidebar.caption(f"Serving model: **{model_name}**")
    inputs = {}
    for feat in features:
        if feat in FEATURE_META:
            label, lo, hi, default = FEATURE_META[feat]
            inputs[feat] = st.sidebar.slider(label, float(lo), float(hi), float(default))
        else:
            inputs[feat] = st.sidebar.number_input(feat, value=0.0)

    st.subheader("User Input")
    input_df = pd.DataFrame([inputs])[features]
    st.dataframe(input_df, use_container_width=True)

    prediction = model.predict(input_df)[0]
    prediction = max(0.0, float(prediction))

    st.subheader("Predicted Wind Power Output (kW)")
    st.success(f"{prediction:,.2f} kW")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**About this model**")
        st.write(
            "Trained via `src/train.py`, tracked in MLflow (experiment "
            "`wind-power-prediction`), and registered in the MLflow Model "
            "Registry as `wind_power_predictor`."
        )
    with col2:
        st.markdown("**Monitoring**")
        st.write(
            "Data drift and stability are monitored continuously with "
            "Evidently AI (`src/monitor.py`) — see `reports/evidently/`."
        )


if __name__ == "__main__":
    main()
