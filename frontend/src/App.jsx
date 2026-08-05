import { useEffect, useState } from "react";
import { getModelInfo, predict } from "./api.js";
import "./App.css";

const TURBINE_STATUS_LABELS = { 0: "Stopped", 1: "Normal", 2: "Cut-out" };

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
      })
      .catch((err) => setLoadError(err.message));
  }, []);

  function handleChange(feature, value) {
    setValues((prev) => ({ ...prev, [feature]: Number(value) }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setPredicting(true);
    setPredictError(null);
    try {
      const res = await predict(values);
      setResult(res);
    } catch (err) {
      setPredictError(err.message);
    } finally {
      setPredicting(false);
    }
  }

  if (loadError) {
    return (
      <div className="page">
        <div className="card error-card">
          <h1>Wind Power Prediction</h1>
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
        <div className="card">
          <p className="muted">Loading model...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>🌬️ Wind Power Prediction</h1>
        <p className="muted">
          MLOps pipeline demo — served live from{" "}
          <strong>{modelInfo.model_name}</strong>, trained on correlation-pruned,
          imputed SCADA features.
        </p>
      </header>

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

          {modelInfo.metrics && Object.keys(modelInfo.metrics).length > 0 && (
            <>
              <h3>Model Performance</h3>
              <table className="metrics-table">
                <tbody>
                  {Object.entries(modelInfo.metrics).map(([k, v]) => (
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

      <footer className="muted footer">
        Trained via <code>src/train.py</code>, tracked in MLflow, monitored for
        drift with Evidently AI — see the project README for the full pipeline.
      </footer>
    </div>
  );
}
