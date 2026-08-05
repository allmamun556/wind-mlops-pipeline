"""Load: write extracted/transformed data into Postgres."""
import pandas as pd
from sqlalchemy import text

from etl.db import get_engine, init_schema


def load_raw_weather(df: pd.DataFrame) -> int:
    """Upsert raw weather observations (idempotent on re-run)."""
    init_schema()
    engine = get_engine()
    rows = df.to_dict(orient="records")
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO raw_weather_observations
                (source, latitude, longitude, observed_at, wind_speed_ms, wind_direction_deg, temperature_c)
            VALUES
                (:source, :latitude, :longitude, :observed_at, :wind_speed_ms, :wind_direction_deg, :temperature_c)
            ON CONFLICT (source, latitude, longitude, observed_at) DO UPDATE SET
                wind_speed_ms = EXCLUDED.wind_speed_ms,
                wind_direction_deg = EXCLUDED.wind_direction_deg,
                temperature_c = EXCLUDED.temperature_c,
                fetched_at = now()
        """), rows)
    return len(rows)


def load_turbine_readings(df: pd.DataFrame) -> int:
    """Upsert the transformed, unified turbine readings table."""
    init_schema()
    engine = get_engine()
    rows = df.to_dict(orient="records")
    if not rows:
        return 0
    columns = [c for c in df.columns]
    col_list = ", ".join(f'"{c}"' if c not in ("source",) else c for c in columns)
    placeholder_list = ", ".join(f":{c}" for c in columns)
    update_list = ", ".join(
        f'"{c}" = EXCLUDED."{c}"' if c not in ("source",) else "source = EXCLUDED.source"
        for c in columns if c not in ("Time", "WTG")
    )
    with engine.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO turbine_readings ({col_list})
            VALUES ({placeholder_list})
            ON CONFLICT ("Time", "WTG") DO UPDATE SET
                {update_list}, loaded_at = now()
        """), rows)
    return len(rows)
