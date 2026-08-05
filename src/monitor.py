"""
Monitoring & Drift Detection (thesis Section 5.3.2.5, Evidently AI)
---------------------------------------------------------------------
Compares a 'reference' window of SCADA data (training-time distribution)
against a 'current' window (most recent production data) and generates:
  1. A Data Summary / Stability report (row/column counts, missing values,
     dtypes) -- mirrors thesis Figures 5.11 / 5.12.
  2. A Data Drift report (per-column drift tests + drift score) -- mirrors
     thesis Figures 5.13 - 5.15.

Both reports are written as standalone HTML (open in any browser) and a
machine-readable JSON summary is written for CI/CD gating (see
.github/workflows/ci-cd.yml), where a pipeline can automatically flag a
build if the share of drifted columns exceeds a threshold.

Usage:
    python src/monitor.py
"""
import json
import os

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

import config

DRIFT_SHARE_THRESHOLD = 0.5  # if >50% of columns drift, flag for retraining


def build_dataset(df: pd.DataFrame) -> Dataset:
    numeric_cols = [c for c in df.select_dtypes(include="number").columns]
    definition = DataDefinition(numerical_columns=numeric_cols)
    return Dataset.from_pandas(df, data_definition=definition)


def run():
    os.makedirs("reports/evidently", exist_ok=True)

    reference = pd.read_csv(config.REFERENCE_DATA_PATH)
    current = pd.read_csv(config.CURRENT_DATA_PATH)
    drop_cols = [c for c in config.NON_FEATURE_COLUMNS if c in reference.columns]
    reference = reference.drop(columns=drop_cols)
    current = current.drop(columns=drop_cols)

    ref_ds = build_dataset(reference)
    cur_ds = build_dataset(current)

    # --- Data Stability / Summary report ---------------------------------
    stability_report = Report([DataSummaryPreset()])
    stability_result = stability_report.run(reference_data=ref_ds, current_data=cur_ds)
    stability_result.save_html("reports/evidently/data_stability_report.html")

    # --- Data Drift report -------------------------------------------------
    drift_report = Report([DataDriftPreset()])
    drift_result = drift_report.run(reference_data=ref_ds, current_data=cur_ds)
    drift_result.save_html("reports/evidently/data_drift_report.html")

    drift_dict = drift_result.dict()
    summary = _summarise_drift(drift_dict)

    with open("reports/evidently/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Reference rows: {len(reference)}  |  Current rows: {len(current)}")
    print(f"Drifted columns: {summary['n_drifted_columns']}/{summary['n_columns']} "
          f"({summary['drift_share']:.1%})")
    print(f"Dataset drift detected: {summary['dataset_drift_detected']}")
    print("Reports written to reports/evidently/*.html")

    if summary["drift_share"] > DRIFT_SHARE_THRESHOLD:
        print(f"\n*** ALERT: drift share {summary['drift_share']:.1%} exceeds "
              f"{DRIFT_SHARE_THRESHOLD:.0%} threshold -> flagging for model retraining. ***")
    # NOTE: this stage always exits 0 so `dvc repro` / the pipeline can still
    # publish the reports. The CI workflow gates on drift_summary.json in a
    # separate step (see .github/workflows/ci-cd.yml) rather than failing
    # this stage outright.
    return 0


def _summarise_drift(drift_dict: dict) -> dict:
    """Extract column-level drift results from an evidently Report.dict()."""
    n_drifted, n_total, share = 0, 0, 0.0
    per_column = {}
    for metric in drift_dict.get("metrics", []):
        name = str(metric.get("metric_name", ""))
        if name.startswith("DriftedColumnsCount"):
            value = metric.get("value", {}) or {}
            n_drifted = int(value.get("count", 0))
            share = float(value.get("share", 0.0))
        elif name.startswith("ValueDrift(column="):
            col = name.split("column=")[1].split(",")[0]
            per_column[col] = metric.get("value")
            n_total += 1

    if n_total == 0:
        n_total = len(pd.read_csv(config.REFERENCE_DATA_PATH).select_dtypes(include="number").columns)
    if n_total and not share:
        share = n_drifted / n_total

    return {
        "n_columns": n_total,
        "n_drifted_columns": n_drifted,
        "drift_share": share,
        "dataset_drift_detected": share > 0.5,
        "per_column_drift_score": per_column,
    }


if __name__ == "__main__":
    raise SystemExit(run())
