# MLflow tracking UI, containerized so it starts with `docker compose up`
# instead of a separately run `mlflow ui` process.
#
# Reads the SAME mlflow.db (run metadata) and mlruns/ (artifacts) that
# `dvc repro` / src/train.py write when run on the host -- training isn't
# containerized in this project, so docker-compose.yml bind-mounts both
# in read-write rather than baking them into the image.
#
# Version pinned to match requirements.txt's mlflow==3.15.1, since the
# SQLite backend-store schema is version-sensitive.
FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==3.15.1
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "0.0.0.0", "--port", "5000"]
