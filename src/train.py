"""
MLOps Workflow - Experimentation & Model Training (thesis Section 5.2/5.3)
---------------------------------------------------------------------------
Trains a suite of regression models to predict ActivePower from turbine
SCADA features, logging every run (params, metrics, artifacts) to MLflow,
then registers the best-performing model in the MLflow Model Registry and
serialises it to disk for the Streamlit serving app.

Two families of models are trained, mirroring the thesis:
  * Classical statistical time-series models (ARIMA, Exponential Smoothing)
    fit chronologically on the target series alone -- included to
    demonstrate (as in the thesis) that univariate time-series models
    under-perform when the true driver (wind speed) isn't modelled.
  * Feature-based regressors (Linear/Polynomial Regression, Decision Tree,
    Random Forest, AdaBoost, Gradient Boosting, MLP/"ANN") fit on the
    correlation-pruned, imputed feature set.

Usage:
    python src/train.py
"""
import json
import time
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import config

warnings.filterwarnings("ignore")


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def load_feature_data():
    train_df = pd.read_csv(config.PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(config.PROCESSED_TEST_PATH)
    with open(config.FEATURE_LIST_PATH) as f:
        features = json.load(f)
    X_train, y_train = train_df[features], train_df[config.TARGET]
    X_test, y_test = test_df[features], test_df[config.TARGET]
    return X_train, X_test, y_train, y_test, features


def load_timeseries_data(n_forecast=2000):
    """Chronological split of the raw target series for ARIMA / Exp. Smoothing."""
    raw = pd.read_csv(config.RAW_DATA_PATH, parse_dates=["Time"]).sort_values("Time")
    series = raw[config.TARGET].interpolate(limit_direction="both")
    train_series = series.iloc[:-n_forecast]
    test_series = series.iloc[-n_forecast:]
    return train_series, test_series


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def get_regressors():
    return {
        "LinearRegression": LinearRegression(),
        "PolynomialRegression": make_pipeline(
            PolynomialFeatures(degree=2), StandardScaler(), LinearRegression()
        ),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, random_state=config.RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=config.RANDOM_STATE
        ),
        "AdaBoost": AdaBoostRegressor(n_estimators=150, random_state=config.RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=config.RANDOM_STATE
        ),
        "ANN_MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=400,
                early_stopping=True, random_state=config.RANDOM_STATE,
            ),
        ),
    }


def run_regressors(X_train, X_test, y_train, y_test, features):
    results = {}
    for name, model in get_regressors().items():
        with mlflow.start_run(run_name=name):
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)

            mlflow.log_param("model_type", name)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_metrics(metrics)
            mlflow.log_metric("train_time_sec", train_time)
            mlflow.sklearn.log_model(model, name="model", input_example=X_train.iloc[:2], serialization_format="pickle")

            results[name] = {"model": model, **metrics}
            print(f"{name:22s} MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}")
    return results


def run_timeseries_models():
    """ARIMA & Exponential Smoothing on the target series alone (no exogenous features)."""
    train_series, test_series = load_timeseries_data()
    results = {}

    # --- ARIMA -------------------------------------------------------
    with mlflow.start_run(run_name="ARIMA"):
        try:
            model = ARIMA(train_series.values, order=(2, 1, 2)).fit()
            forecast = model.forecast(steps=len(test_series))
        except Exception as e:
            print(f"ARIMA fit issue ({e}); falling back to simple order (1,1,1)")
            model = ARIMA(train_series.values, order=(1, 1, 1)).fit()
            forecast = model.forecast(steps=len(test_series))
        metrics = evaluate(test_series.values, forecast)
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("order", "(2,1,2)")
        mlflow.log_metrics(metrics)
        results["ARIMA"] = {"model": model, **metrics}
        print(f"{'ARIMA':22s} MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}")

    # --- Exponential Smoothing ----------------------------------------
    with mlflow.start_run(run_name="ExponentialSmoothing"):
        model = ExponentialSmoothing(
            train_series.values, trend="add", seasonal=None
        ).fit()
        forecast = model.forecast(len(test_series))
        metrics = evaluate(test_series.values, forecast)
        mlflow.log_param("model_type", "ExponentialSmoothing")
        mlflow.log_metrics(metrics)
        results["ExponentialSmoothing"] = {"model": model, **metrics}
        print(f"{'ExponentialSmoothing':22s} MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.4f}")

    return results


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    print("Loading data ...")
    X_train, X_test, y_train, y_test, features = load_feature_data()

    print("\n=== Training feature-based regressors ===")
    reg_results = run_regressors(X_train, X_test, y_train, y_test, features)

    print("\n=== Training classical time-series models (target-only) ===")
    ts_results = run_timeseries_models()

    all_results = {**reg_results, **ts_results}
    comparison = pd.DataFrame(
        {k: {m: v[m] for m in ["MAE", "MSE", "RMSE", "R2"]} for k, v in all_results.items()}
    ).T.sort_values("R2", ascending=False)
    comparison.to_csv("reports/model_comparison.csv")
    print("\n=== Model Comparison (sorted by R2) ===")
    print(comparison.round(4))

    best_name = comparison.index[0]
    best_model = all_results[best_name]["model"]
    print(f"\nBest model: {best_name} (R2={comparison.loc[best_name, 'R2']:.4f})")

    # Register best model (only feature-based models are servable via the app)
    if best_name in reg_results:
        with mlflow.start_run(run_name=f"BEST_{best_name}"):
            mlflow.log_param("selected_as_best", True)
            mlflow.log_metrics({m: comparison.loc[best_name, m] for m in ["MAE", "MSE", "RMSE", "R2"]})
            mlflow.sklearn.log_model(
                best_model, name="model",
                registered_model_name=config.MODEL_REGISTRY_NAME,
                input_example=X_train.iloc[:2],
                serialization_format="pickle",
            )
        joblib.dump({"model": best_model, "features": features, "name": best_name},
                    config.BEST_MODEL_PATH)
        print(f"Best model saved to {config.BEST_MODEL_PATH} and registered in MLflow Model Registry "
              f"as '{config.MODEL_REGISTRY_NAME}'.")
    else:
        # fall back to best feature-based model for serving purposes
        best_feature_name = comparison.loc[comparison.index.isin(reg_results.keys())].index[0]
        best_feature_model = reg_results[best_feature_name]["model"]
        joblib.dump({"model": best_feature_model, "features": features, "name": best_feature_name},
                    config.BEST_MODEL_PATH)
        print(f"(Best overall model '{best_name}' is a target-only time-series model and isn't "
              f"servable from feature inputs; serving '{best_feature_name}' instead.)")

    return comparison


if __name__ == "__main__":
    main()
