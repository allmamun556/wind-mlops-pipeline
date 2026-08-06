# Multi-stage build: compile the Vite/React app, then serve the static
# bundle with Nginx.
#
# VITE_API_URL is a build ARG (not hardcoded) so this same Dockerfile works
# two ways:
#   - docker-compose (local/production-stack): defaults to same-origin
#     "/api", which nginx.conf reverse-proxies to the "backend" service.
#   - standalone Render deploy: set the VITE_API_URL environment variable
#     in Render's dashboard to the backend service's public URL (Render
#     auto-passes dashboard env vars as build args of the same name) and
#     the frontend calls it directly over CORS instead of a proxy.
FROM node:20-slim AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
# Cross-links to the other dashboards in the stack (header nav). Defaults
# match docker-compose.yml's local ports; override as build args (Render:
# dashboard env vars of the same name) for a standalone deploy.
ARG VITE_GRAFANA_URL=http://localhost:3001/d/wind-power-monitoring/wind-power-prediction-e28094-model-monitoring?orgId=1&refresh=10s
ARG VITE_PROMETHEUS_URL=http://localhost:9090
ARG VITE_MLFLOW_URL=http://localhost:5000/#/experiments/1/runs?workflowType=machine_learning&searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D&compareRunsMode=TABLE
ENV VITE_GRAFANA_URL=$VITE_GRAFANA_URL
ENV VITE_PROMETHEUS_URL=$VITE_PROMETHEUS_URL
ENV VITE_MLFLOW_URL=$VITE_MLFLOW_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK CMD wget -qO- http://127.0.0.1:80/ || exit 1
