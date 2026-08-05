"""ETL entrypoint: real weather APIs -> Postgres -> CSV snapshot for DVC.

Usage:
    python src/etl/run_etl.py --backfill-days 730

Requires a reachable Postgres (see docker-compose.yml's `postgres` service,
or set DATABASE_URL). Extracts real weather from two Open-Meteo sources,
derives the full turbine SCADA schema, loads both the raw and transformed
data into Postgres, then exports a CSV snapshot so the existing DVC
pipeline (preprocessing -> train -> monitor) can consume it unchanged.
"""
import argparse
import os
import sys

# src/etl/run_etl.py is run directly (python src/etl/run_etl.py), so Python
# only puts src/etl/ on sys.path by default. The rest of the pipeline's
# scripts assume src/ itself is on the path (e.g. `import config` from
# src/train.py) -- add it explicitly so `import config`, `import etl.*`,
# and `from data_generation import power_curve` all resolve consistently.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402
from etl import extract, transform  # noqa: E402
from etl.load import load_raw_weather, load_turbine_readings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Real-weather ETL: Open-Meteo -> Postgres -> CSV")
    parser.add_argument("--backfill-days", type=int, default=730,
                         help="Days of historical weather to backfill (default: ~2 years)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=config.ETL_RAW_DATA_PATH)
    args = parser.parse_args()

    print(f"Extracting real weather for ({config.WEATHER_LAT}, {config.WEATHER_LON}) "
          f"[{args.backfill_days} days backfill + forecast] ...")
    weather = extract.extract_all(config.WEATHER_LAT, config.WEATHER_LON, backfill_days=args.backfill_days)
    print(f"  {len(weather):,} real observations pulled "
          f"({(weather['source'] == 'open-meteo-historical').sum():,} historical, "
          f"{(weather['source'] == 'open-meteo-forecast').sum():,} forecast)")

    print("Loading raw weather into Postgres (raw_weather_observations) ...")
    n_raw = load_raw_weather(weather)
    print(f"  {n_raw:,} rows upserted")

    print("Transforming into unified turbine schema ...")
    turbine_df = transform.derive_turbine_readings(weather, seed=args.seed)
    print(f"  {turbine_df.shape[0]:,} rows x {turbine_df.shape[1]} columns")

    print("Loading transformed readings into Postgres (turbine_readings) ...")
    n_turbine = load_turbine_readings(turbine_df)
    print(f"  {n_turbine:,} rows upserted")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    turbine_df.drop(columns=["source"]).to_csv(args.out, index=False)
    print(f"CSV snapshot written -> {args.out}")


if __name__ == "__main__":
    main()
