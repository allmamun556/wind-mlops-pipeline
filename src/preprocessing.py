"""
Data Modeling / Preprocessing (thesis Section 5.2)
---------------------------------------------------
1. Load raw SCADA data
2. Plot & report correlation matrix (before/after)
3. Drop highly-correlated redundant features
4. Impute missing values per-column strategy
5. Train/test split, write processed CSVs
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import config


def load_raw(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Time"])
    return df


def plot_correlation(df: pd.DataFrame, path: str, title: str):
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="rocket", annot_kws={"size": 6})
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return corr


def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, strategy in config.IMPUTATION_STRATEGY.items():
        if col not in df.columns:
            continue
        if strategy == "interpolate":
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
    # Safety net for any remaining NaNs
    df = df.fillna(df.median(numeric_only=True))
    return df


def run(raw_path: str = config.RAW_DATA_PATH):
    print(f"Loading raw data from {raw_path} ...")
    df = load_raw(raw_path)
    print(f"Raw shape: {df.shape}")

    print("Missing values per column:")
    print(df.isna().sum())

    print("Plotting correlation matrix (before feature removal) ...")
    plot_correlation(df, "reports/figures/correlation_before.png",
                      "Correlation Plot (before removing correlated features)")

    print(f"Dropping highly-correlated columns: {config.DROP_HIGH_CORRELATION}")
    df = df.drop(columns=[c for c in config.DROP_HIGH_CORRELATION if c in df.columns])

    print("Plotting correlation matrix (after feature removal) ...")
    plot_correlation(df, "reports/figures/correlation_after.png",
                      "Correlation Plot (after removing correlated features)")

    print("Imputing missing values ...")
    df = impute(df)
    assert df.isna().sum().sum() == 0, "Missing values remain after imputation!"

    # Persist a monitoring reference / current split before further splitting
    os.makedirs("data/monitoring", exist_ok=True)
    split_idx = int(len(df) * 0.85)
    df.iloc[:split_idx].to_csv(config.REFERENCE_DATA_PATH, index=False)
    df.iloc[split_idx:].to_csv(config.CURRENT_DATA_PATH, index=False)

    feature_cols = [c for c in df.columns
                     if c not in config.NON_FEATURE_COLUMNS + [config.TARGET]]
    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, shuffle=True
    )
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv(config.PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(config.PROCESSED_TEST_PATH, index=False)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print("Preprocessing complete.")
    return train_df, test_df, feature_cols


if __name__ == "__main__":
    run()
