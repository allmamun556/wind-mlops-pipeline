import { useEffect, useState } from "react";
import { getModelInfo, predict } from "./api.js";
import "./App.css";

const TURBINE_STATUS_LABELS = { 0: "Stopped", 1: "Normal", 2: "Cut-out" };

const DASHBOARD_LINKS = [
  {
    label: "Grafana",
    url:
      import.meta.env.VITE_GRAFANA_URL ||
      "http://localhost:3001/d/wind-power-monitoring/wind-power-prediction-e28094-model-monitoring?orgId=1&refresh=10s",
  },
  { label: "Prometheus", url: import.meta.env.VITE_PROMETHEUS_URL || "http://localhost:9090" },
  {
    label: "MLflow",
    url:
      import.meta.env.VITE_MLFLOW_URL ||
      "http://localhost:5000/#/experiments/1/runs?workflowType=machine_learning&searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D&compareRunsMode=TABLE",
  },
  { label: "GitHub", url: "https://github.com/allmamun556/wind-mlops-pipeline" },
  { label: "DagsHub", url: "https://dagshub.com/allmamun556/wind-mlops-pipeline" },
];

function Header({ children }) {
  return (
    <header className="site-header">
      <div className="site-header-top">
        <div className="brand">
          <span aria-hidden="true">🌬️</span>
          <h1>Wind Power Prediction</h1>
        </div>
        <nav className="dashboard-nav">
          {DASHBOARD_LINKS.map(({ label, url }) => (
            <a key={label} href={url} target="_blank" rel="noreferrer">
              {label} ↗
            </a>
          ))}
        </nav>
      </div>
      {children && <p className="muted site-tagline">{children}</p>}
    </header>
  );
}

export default function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState(null);

  useEffect(() => {
    getModelInfo()
      .then((info) => {
        setModelInfo(info);
        const defaults = {};
        for (const feat of info.features) {
          defaults[feat] = info.feature_meta[feat]?.default ?? 0;
        }
        setValues(defaults);
        runPrediction(defaults);
      })
      .catch((err) => setLoadError(err.message));
  }, []);

  function handleChange(feature, value) {
    setValues((prev) => ({ ...prev, [feature]: Number(value) }));
  }

  async function runPrediction(vals) {
    setPredicting(true);
    setPredictError(null);
    try {
      const res = await predict(vals);
      setResult(res);
    } catch (err) {
      setPredictError(err.message);
    } finally {
      setPredicting(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    runPrediction(values);
  }

  if (loadError) {
    return (
      <div className="page">
        <Header />
        <div className="card error-card">
          <p className="error-text">Could not reach the prediction API: {loadError}</p>
          <p className="muted">
            Make sure the FastAPI backend is running, e.g.{" "}
            <code>uvicorn backend.main:app --port 8000</code>.
          </p>
        </div>
      </div>
    );
  }

  if (!modelInfo) {
    return (
      <div className="page">
        <Header />
        <div className="card">
          <p className="muted">Loading model...</p>
        </div>
      </div>
    );
  }

  const { R2, ...otherMetrics } = modelInfo.metrics || {};

  return (
    <div className="page">
      <Header>
        MLOps pipeline demo — served live from <strong>{modelInfo.model_name}</strong>,
        trained on correlation-pruned, imputed SCADA features.
      </Header>

      <div className="layout">
        <form className="card" onSubmit={handleSubmit}>
          <h2>Turbine Operating Conditions</h2>
          <div className="field-grid">
            {modelInfo.features.map((feat) => {
              const meta = modelInfo.feature_meta[feat] || {};
              const value = values[feat] ?? 0;

              if (feat === "TurbineStatus") {
                return (
                  <label className="field" key={feat}>
                    <span className="field-label">{meta.label || feat}</span>
                    <select
                      value={value}
                      onChange={(e) => handleChange(feat, e.target.value)}
                    >
                      {[0, 1, 2].map((v) => (
                        <option value={v} key={v}>
                          {TURBINE_STATUS_LABELS[v]}
                        </option>
                      ))}
                    </select>
                  </label>
                );
              }

              return (
                <label className="field" key={feat}>
                  <span className="field-label">
                    {meta.label || feat}
                    <span className="field-value">
                      {value}
                      {meta.unit ? ` ${meta.unit}` : ""}
                    </span>
                  </span>
                  <input
                    type="range"
                    min={meta.min ?? 0}
                    max={meta.max ?? 100}
                    step={0.1}
                    value={value}
                    onChange={(e) => handleChange(feat, e.target.value)}
                  />
                </label>
              );
            })}
          </div>

          <button type="submit" disabled={predicting}>
            {predicting ? "Predicting..." : "Predict Power Output"}
          </button>
          {predictError && <p className="error-text">{predictError}</p>}
        </form>

        <aside className="card result-card">
          <h2>Prediction</h2>
          {result ? (
            <>
              <div className="prediction-value">
                {result.prediction.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                <span className="prediction-unit"> {result.unit}</span>
              </div>
              <p className="muted">Predicted active power output.</p>
            </>
          ) : (
            <p className="muted">Adjust the inputs and click Predict.</p>
          )}

          {R2 !== undefined && (
            <div className="headline-metric">
              <span className="headline-label">Model R²</span>
              <span className="headline-value">{R2.toFixed(4)}</span>
            </div>
          )}

          {Object.keys(otherMetrics).length > 0 && (
            <>
              <h3>Model Performance</h3>
              <table className="metrics-table">
                <tbody>
                  {Object.entries(otherMetrics).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{v.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </aside>
      </div>

      <section className="card about-card">
        <h2>About this project</h2>
        <p className="muted">
          A working implementation of the MLOps pipeline from the M.Sc. thesis{" "}
          <em>&ldquo;Modern ML-CI/CD, Experiment Tracking and Monitoring for Wind
          Power Prediction Data&rdquo;</em> (BHT Berlin, 2024) — rebuilt end-to-end
          as a runnable demo, not just scaffolding.
        </p>
        <ol className="pipeline-steps">
          <li>
            <strong>Data</strong>
            <span>synthetic SCADA generation, correlation-pruned &amp; imputed</span>
          </li>
          <li>
            <strong>Train</strong>
            <span>9 models compared, MLflow-tracked, best model registered</span>
          </li>
          <li>
            <strong>Monitor</strong>
            <span>Evidently AI drift detection, gated in CI/CD</span>
          </li>
          <li>
            <strong>Serve</strong>
            <span>this page (FastAPI + React), plus a Streamlit app</span>
          </li>
        </ol>
        <p className="muted">
          Full architecture, results, and source:{" "}
          <a
            href="https://github.com/allmamun556/wind-mlops-pipeline#readme"
            target="_blank"
            rel="noreferrer"
          >
            project README ↗
          </a>
        </p>
      </section>

      <footer className="muted footer">Wind Power MLOps Pipeline — A. A. Mamun</footer>
    </div>
  );
}
