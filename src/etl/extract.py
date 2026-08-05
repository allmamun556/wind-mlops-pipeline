"""Extract: pull real weather data from two distinct Open-Meteo sources.

Open-Meteo (https://open-meteo.com) is free, requires no API key, and
covers both historical backfill and live forecast data — a genuine
two-source ETL without needing paid/gated APIs.

  - Historical Weather API: hourly observations back to 1940, used to
    backfill training data.
  - Forecast API: current/near-term conditions, simulating what a
    production system would ingest incrementally going forward.
"""
from datetime import date, timedelta

import pandas as pd
import requests

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_FIELDS = "wind_speed_10m,wind_direction_10m,temperature_2m"


def _to_dataframe(payload: dict, source: str, lat: float, lon: float) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({
        "observed_at": pd.to_datetime(hourly["time"]),
        "wind_speed_ms": hourly["wind_speed_10m"],
        "wind_direction_deg": hourly["wind_direction_10m"],
        "temperature_c": hourly["temperature_2m"],
    })
    df["source"] = source
    df["latitude"] = lat
    df["longitude"] = lon
    return df


def fetch_historical(lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Real hourly weather observations for [start_date, end_date]."""
    resp = requests.get(HISTORICAL_URL, params={
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": HOURLY_FIELDS,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }, timeout=30)
    resp.raise_for_status()
    return _to_dataframe(resp.json(), source="open-meteo-historical", lat=lat, lon=lon)


def fetch_forecast(lat: float, lon: float, forecast_days: int = 2) -> pd.DataFrame:
    """Real current + near-term forecast weather observations."""
    resp = requests.get(FORECAST_URL, params={
        "latitude": lat,
        "longitude": lon,
        "forecast_days": forecast_days,
        "hourly": HOURLY_FIELDS,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }, timeout=30)
    resp.raise_for_status()
    return _to_dataframe(resp.json(), source="open-meteo-forecast", lat=lat, lon=lon)


def extract_all(lat: float, lon: float, backfill_days: int = 730) -> pd.DataFrame:
    """Both sources combined: historical backfill + live forecast."""
    end = date.today() - timedelta(days=5)  # historical API lags ~5 days
    start = end - timedelta(days=backfill_days)

    historical = fetch_historical(lat, lon, start, end)
    forecast = fetch_forecast(lat, lon)

    combined = pd.concat([historical, forecast], ignore_index=True)
    combined = combined.drop_duplicates(subset=["source", "latitude", "longitude", "observed_at"])
    return combined
