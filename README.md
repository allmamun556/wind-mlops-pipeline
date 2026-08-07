# Wind Power MLOps Pipeline

[![ML-CI/CD](https://github.com/allmamun556/wind-mlops-pipeline/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/allmamun556/wind-mlops-pipeline/actions/workflows/ci-cd.yml)
[![CML Report](https://github.com/allmamun556/wind-mlops-pipeline/actions/workflows/cml.yml/badge.svg)](https://github.com/allmamun556/wind-mlops-pipeline/actions/workflows/cml.yml)
[![DVC](https://img.shields.io/badge/DVC-tracked-13ADC7?logo=dvc&logoColor=white)](https://dvc.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-Project-blue?logo=dagshub)](https://dagshub.com/allmamun556/wind-mlops-pipeline)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Grafana](https://img.shields.io/badge/Grafana-dashboard-F46800?logo=grafana&logoColor=white)](https://grafana.com)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io)
[![React](https://img.shields.io/badge/React-frontend-61DAFB?logo=react&logoColor=black)](https://react.dev)

A working implementation of the MLOps pipeline designed in *"Modern ML-CI/CD,
Experiment Tracking and Monitoring for Wind Power Prediction Data"*
(A. A. Mamun, M.Sc. Thesis, BHT Berlin, 2024) — rebuilt end-to-end as a
runnable portfolio project: synthetic SCADA data → correlation-pruned
preprocessing → multi-model training with MLflow tracking → Evidently AI
drift monitoring → CI/CD via GitHub Actions → CML reporting → served two
ways: the thesis-original Streamlit app, and a production-style FastAPI +
React stack — with a live Grafana + Prometheus monitoring dashboard on
top (see [Production serving](#production-serving-fastapi--react) and
[Model monitoring](#model-monitoring-grafana--prometheus)).

Every stage below was actually executed (not just scaffolded) — see
`reports/` for real output: correlation plots, a 9-model comparison table,
and live Evidently AI drift/stability HTML reports with genuine detected
drift.

## Architecture

Phases ①–④ are adapted from the thesis's Google MLOps →
open-source-tools mapping (Chapter 4, Figures 4.1–4.3). Phases ⑤–⑥ are
the production serving + live monitoring stack layered on top — not part
of the original thesis design, added here so "monitoring" is a live
dashboard instead of static HTML reports:

```mermaid
flowchart LR
    subgraph P1["① Data Ingestion &amp; Versioning"]
        GEN["data_generation.py<br/>synthetic SCADA, seed=42"] --> RAW[("data/raw/*.csv")]
        RAW --> LOCK["dvc.yaml / dvc.lock"]
    end

    subgraph P2["② Experimentation &amp; Training"]
        PREP["preprocessing.py<br/>correlation pruning + imputation"] --> TRAIN["train.py<br/>9 models compared"]
        TRAIN --> MODEL[("best_model.pkl")]
        TRAIN -.->|"logs every run"| MLFLOW[("MLflow Tracking<br/>+ Model Registry")]
    end

    subgraph P3["③ CI/CD — GitHub Actions"]
        TEST["pytest tests/"] --> REPRO["dvc repro"]
        REPRO --> GATE{"drift share<br/>&gt; 50%?"}
        GATE -->|"warn (non-blocking)"| DEPLOY["deploy job<br/>Render hook"]
    end

    subgraph P4["④ Monitoring &amp; Feedback"]
        EVID["monitor.py<br/>Evidently AI"] --> DRIFT[("drift_summary.json<br/>+ HTML reports")]
        DRIFT --> CML["CML report<br/>commit comment"]
    end

    subgraph P5["⑤ Model Serving"]
        STREAMLIT["Streamlit app.py<br/>→ HF Space"]
        BACKEND["FastAPI backend<br/>/predict /model-info"]
        FRONTEND["React frontend<br/>prediction UI"]
        BACKEND --> FRONTEND
    end

    subgraph P6["⑥ Live Monitoring"]
        PROM[("Prometheus<br/>scrapes /metrics every 10s")]
        GRAFANA["Grafana dashboard"]
        PROM --> GRAFANA
    end

    LOCK --> PREP
    LOCK <-->|"push / pull"| DAGSHUB[("DagsHub<br/>DVC remote + hosted MLflow")]
    REPRO -.->|triggers| EVID
    MODEL --> STREAMLIT
    MODEL --> BACKEND
    DEPLOY -.->|deploys| STREAMLIT
    DEPLOY -.->|deploys| BACKEND
    BACKEND -->|"serving metrics"| PROM
    DRIFT -.->|"re-read on scrape"| PROM

    FRONTEND ---|"nav links out to"| HUB{{"Grafana · Prometheus · MLflow<br/>GitHub · DagsHub"}}
    GRAFANA ---|"links back to"| HUB

    P1 ~~~ P2 ~~~ P3 ~~~ P4 ~~~ P5 ~~~ P6
```

See [Production serving](#production-serving-fastapi--react) and
[Model monitoring](#model-monitoring-grafana--prometheus) below for the
detail behind phases ⑤–⑥ (env vars, ports, what each panel shows).

| Thesis tool | Role here | Where |
|---|---|---|
| Git + GitHub | Code/config version control | this repo |
| Git LFS | Large file versioning | `.gitattributes` |
| DVC | Data + pipeline versioning, reproducibility | `dvc.yaml`, `dvc.lock` |
| MLflow Tracking + Model Registry | Experiment logging, model versioning | `src/train.py`, `mlflow.db` |
| GitHub Actions | CI/CD automation | `.github/workflows/ci-cd.yml` |
| CML | Automated PR performance reports | `.github/workflows/cml.yml` |
| Evidently AI | Data drift & stability computation | `src/monitor.py`, `reports/evidently/` |
| Prometheus + Grafana | Live model monitoring dashboard | `backend/metrics.py`, `monitoring/` |
| Streamlit | Model serving UI (thesis-original) | `app.py` |
| FastAPI + React | Production serving API + SPA | `backend/`, `frontend/` |
| Docker / Render | Deployment target | `Dockerfile`, deploy job in `ci-cd.yml` |

## What's real vs. what you'd swap in for production

This is a **portfolio/demo implementation** built without ENERTRAG
credentials, so a few things are stand-ins by design:

- **Data**: synthetic SCADA data (`src/data_generation.py`) that reproduces
  the thesis's feature set, correlation structure, missing-value rates, and
  a deliberately-injected drift segment — so the drift detector has
  something genuine to catch. Swap in real ENERTRAG SCADA exports (or the
  Kaggle/Enerjisa datasets used in other notebooks) by pointing
  `config.RAW_DATA_PATH` at a CSV with the same column names; nothing else
  needs to change.
- **MLflow / DVC backends**: local SQLite + local filesystem (`mlflow.db`,
  `.dvc/`). For a team setting, point `MLFLOW_TRACKING_URI` at a hosted
  MLflow server and add a DVC remote (`dvc remote add -d storage s3://...`).
- **Deploy step**: the GitHub Actions `deploy` job pushes to Render via a
  deploy hook — set the `RENDER_DEPLOY_HOOK` secret in the GitHub repo to
  enable it (see Render's dashboard → service → Settings → Deploy Hook).

## Where to see the automation run

Everything below happens on every push to `main` (and nightly on a cron) —
no need to dig through raw logs to see it:

- **Unit tests** (`tests/test_pipeline.py`): the `test` job in
  [`ci-cd.yml`](.github/workflows/ci-cd.yml) runs `pytest -v` and writes a
  pass/fail summary to the run's **Summary** tab (Actions → pick a run →
  top of the page), in addition to the full `-v` output in the job log.
- **CI/CD pipeline status**: the badges at the top of this README link
  straight to the [Actions tab](https://github.com/allmamun556/wind-mlops-pipeline/actions)
  — green means the latest `dvc repro` (data → train → monitor) and drift
  gate passed.
- **CML model performance report**: [`cml.yml`](.github/workflows/cml.yml)
  posts the 9-model comparison table + drift summary two places every
  push — as a **comment on the commit** (see any commit's page on GitHub)
  and on the workflow run's **Summary** tab, so it's visible without
  opening a diff.
- **Pipeline artifacts** (reports, best model, mlflow.db): uploaded by the
  `pipeline` job as a downloadable build artifact on each run (Actions →
  run → Artifacts section at the bottom).

## Results (this run)

Trained 9 models (7 feature-based regressors + 2 univariate time-series
baselines) on 40,000 rows of synthetic SCADA data, 14 features after
correlation pruning:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **RandomForest** (best, registered) | 35.33 | 113.25 | **0.9722** |
| GradientBoosting | 36.09 | 113.74 | 0.9719 |
| ANN (MLPRegressor) | 41.52 | 115.02 | 0.9713 |
| PolynomialRegression | 53.68 | 123.46 | 0.9669 |
| LinearRegression | 61.45 | 134.56 | 0.9607 |
| DecisionTree | 41.22 | 148.63 | 0.9521 |
| AdaBoost | 229.16 | 291.96 | 0.8152 |
| ARIMA (target-only) | 235.36 | 455.12 | -0.0917 |
| ExponentialSmoothing (target-only) | 232.82 | 458.08 | -0.1060 |

RandomForest and GradientBoosting are close enough (R² 0.9722 vs. 0.9719)
that which one wins "best" can flip between reproductions —
`RandomForestRegressor` here uses `n_jobs=-1` (parallel), and parallel
floating-point reduction isn't bit-identical run-to-run even with a fixed
`random_state`. Every other model is single-threaded and fully
deterministic given the same seed.

This reproduces the thesis's core qualitative finding: tree ensembles and
neural nets dominate, while univariate time-series models fail because they
never see the actual driver of power output (wind speed) — see
`reports/figures/model_comparison_chart.png`.

**Drift monitoring**: an artificial distribution shift was injected into the
last 15% of the series (ambient temperature +4°C, gearbox oil +6°C, wind
speed ×0.85, generator winding +5°C). Evidently AI correctly flagged
**13 of 15 monitored columns (86.7%) as drifted**, correctly exceeding the
50% dataset-drift threshold — see `reports/evidently/drift_summary.json`
and the full HTML reports.

### Baseline vs. pipeline (thesis Table 6.1 style)

| Metric | Manual/baseline workflow | This pipeline |
|---|---|---|
| Deployment | Manual scripts, ad hoc | `dvc repro` + GitHub Actions, one command |
| Reproducibility | Spreadsheets, tribal knowledge | MLflow (9 tracked runs) + DVC pipeline lock file |
| Monitoring | Manual/periodic checks | Evidently AI, automatic drift gating in CI |
| Versioning | Code only | Code (Git) + data (DVC/Git LFS) + models (MLflow Registry) |
| Reporting | Manual write-ups | CML auto-posts metrics to every PR |

## Project layout

```
wind-mlops-pipeline/
├── src/
│   ├── config.py          # column lists, paths, hyperparameters
│   ├── data_generation.py # synthetic SCADA data (swap for real data)
│   ├── preprocessing.py   # correlation pruning + imputation
│   ├── train.py           # 9-model MLflow-tracked training + registry
│   └── monitor.py         # Evidently AI drift/stability reports
├── app.py                 # Streamlit serving UI (thesis-original, HF Space deploy)
├── backend/                # FastAPI production serving API
│   ├── main.py             # /health, /model-info, /predict
│   ├── requirements.txt    # lean serving-only deps (no dvc/mlflow/evidently)
│   └── Dockerfile
├── frontend/                # React (Vite) production UI, calls backend/
│   ├── src/App.jsx          # prediction form + result/metrics panel
│   ├── nginx.conf           # serves the build + proxies /api -> backend
│   └── Dockerfile
├── monitoring/              # Grafana + Prometheus model monitoring stack
│   ├── prometheus/prometheus.yml       # scrapes backend:8000/metrics
│   └── grafana/
│       ├── provisioning/               # auto-wired datasource + dashboard
│       └── dashboards/wind_power_monitoring.json
├── docker-compose.yml      # backend + frontend + prometheus + grafana
├── dvc.yaml / dvc.lock    # reproducible pipeline definition
├── .github/workflows/
│   ├── ci-cd.yml           # test → dvc repro → drift gate → deploy
│   └── cml.yml             # PR performance comment bot
├── tests/test_pipeline.py # unit tests (run in CI)
├── reports/
│   ├── model_comparison.csv
│   ├── figures/            # correlation + model comparison plots
│   └── evidently/          # data_drift_report.html, data_stability_report.html
├── models/best_model.pkl  # serialized best model + feature list
├── mlflow.db               # MLflow tracking store (sqlite)
├── Dockerfile               # Streamlit image (HF Space deploy)
└── requirements.txt
```

## Running it yourself

```bash
pip install -r requirements.txt

# Full pipeline, one command (data -> preprocess -> train -> monitor)
dvc repro

# Inspect experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000

# Serve predictions
streamlit run app.py                                 # http://localhost:8501

# Run tests (same as CI)
pytest tests/ -v
```

## Production serving: FastAPI + React

In addition to the thesis-original Streamlit app (`app.py`), the model is
also served via a decoupled FastAPI backend and React frontend — a more
typical production shape (JSON API + SPA) than a Streamlit script.

**Local development** (two terminals, backend then frontend):

```bash
# Terminal 1 — API on http://localhost:8000 (docs at /docs)
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — UI on http://localhost:5173, proxies to the API above
cd frontend
npm install
npm run dev
```

**Production (Docker Compose)** — builds a lean backend image (model +
FastAPI only, no dvc/mlflow/evidently) and an Nginx-served static frontend
that reverse-proxies `/api/*` to the backend, so no CORS setup or
hardcoded backend URL is needed in the built JS:

```bash
docker compose up --build
# UI:  http://localhost:8082
# API: http://localhost:8082/api/health, /api/model-info, /api/predict
```

The backend reads whatever `models/best_model.pkl` the `train` DVC stage
last produced — retrain (`dvc repro`) and rebuild the backend image to
serve an updated model.

## Model monitoring: Grafana + Prometheus

Rather than opening static Evidently AI HTML reports by hand, model and
drift monitoring is also available as a live dashboard app. Evidently AI
still does the actual drift computation in `src/monitor.py` (unchanged —
no reason to reimplement statistical drift detection); what's new is the
presentation layer (see phases ⑤–⑥ of the
[architecture diagram](#architecture) above for how this fits together):

- **`backend/metrics.py`** exposes a Prometheus `/metrics` endpoint on
  the FastAPI backend with two kinds of signal:
  - **Live serving metrics** (update in real time as requests come in):
    prediction request rate, predicted power output, HTTP latency/status
    per endpoint.
  - **Pipeline metrics** (re-read from disk on every scrape, so they
    reflect the latest `dvc repro` run without a backend restart):
    held-out model performance (`reports/model_comparison.csv`) and the
    latest Evidently drift report (`reports/evidently/drift_summary.json`
    — dataset drift flag, drift share, per-column drift scores).
- **Prometheus** (`monitoring/prometheus/prometheus.yml`) scrapes that
  endpoint every 10s.
- **Grafana** (`monitoring/grafana/`) is pre-provisioned on startup with
  the Prometheus datasource and a ready-made dashboard — nothing to
  click together by hand.

```bash
docker compose up --build
# Dashboard: http://localhost:3001  (anonymous viewer access, no login)
# Prometheus: http://localhost:9090
# MLflow: http://localhost:5000 (reads the same mlflow.db / mlruns/ as a
# local `mlflow ui` would -- see monitoring/mlflow/Dockerfile)
```

The dashboard (`monitoring/grafana/dashboards/wind_power_monitoring.json`)
shows: dataset drift status, drift share gauge (50% gate threshold, same
as the CI drift gate), per-column drift scores, model R²/MAE/RMSE,
live predicted power output, prediction request rate, and API latency —
so making predictions through the React frontend at `:8082` visibly moves
the dashboard in real time.

**Cross-linking between dashboards**: the React frontend's header has nav
links out to Grafana, Prometheus, MLflow, GitHub, and DagsHub
(`frontend/.env.example` — `VITE_GRAFANA_URL` / `VITE_PROMETHEUS_URL` /
`VITE_MLFLOW_URL`, defaulting to their local ports; override for a
standalone deploy), and the Grafana dashboard has links back to the
frontend, Prometheus, MLflow, GitHub, and DagsHub via its `links` field —
so you can jump between all six without retyping URLs.

> Anonymous viewer access is enabled for convenience in this portfolio
> demo (`GF_AUTH_ANONYMOUS_ENABLED=true` in `docker-compose.yml`) — turn
> that off and set a real admin password before exposing this beyond
> localhost.

## Mapping back to the thesis hypotheses

- **H1 (reproducibility/transparency via MLflow + monitoring tooling)**:
  every one of the 9 model runs here is fully logged (params, metrics,
  model artifact) and re-creatable with `dvc repro` from a clean checkout —
  `dvc.lock` pins exact command + dependency hashes per stage.
- **H2 (continuous monitoring mitigates data drift impact)**: demonstrated
  directly — the injected drift was caught at 86.7% column drift share,
  well above the 50% gating threshold, which is exactly the kind of signal
  that would trigger a retrain in the CI workflow's drift-gate step.
- **H3 (CI/CD cuts deployment time/effort without sacrificing stability)**:
  the entire data→train→monitor→(deploy) sequence runs as a single
  `dvc repro` / GitHub Actions invocation instead of manual steps, and
  8 of 9 models are exactly reproducible run-to-run (fixed seed,
  single-threaded). The exception — `RandomForestRegressor`'s `n_jobs=-1`
  parallelism — is a known, bounded source of run-to-run float variance,
  not evidence against reproducibility (see the
  [Results](#results-this-run) note above).

## Future work (extending the thesis's own "Future Work" chapter)

- Swap in real ENERTRAG turbine SCADA exports and retrain.
- Add CNN/RNN/LSTM sequence models (skipped here to keep the build fast —
  sklearn/statsmodels only); thesis reports LSTM at R²≈0.90, below the tree
  ensembles, so the expected uplift is limited for this feature set.
- Point DVC/MLflow at shared remotes (S3/Azure Blob + hosted MLflow) for
  team use instead of local SQLite/filesystem.
- `RENDER_DEPLOY_HOOK` is wired up in the GitHub Actions `deploy` job —
  set it as a repo secret once the Render service exists.
