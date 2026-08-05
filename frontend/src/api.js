const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request to ${path} failed with ${res.status}`);
  }
  return res.json();
}

export function getModelInfo() {
  return request("/model-info");
}

export function predict(features) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ features }),
  });
}
