# Lean production image for the FastAPI serving layer (backend/main.py).
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY models/ models/
COPY reports/model_comparison.csv reports/model_comparison.csv
COPY reports/evidently/drift_summary.json reports/evidently/drift_summary.json

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
