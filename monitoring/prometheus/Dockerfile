# Standalone Prometheus image for a direct Render deploy (separate from the
# docker-compose stack, which instead bind-mounts prometheus.yml directly).
#
# Prometheus's YAML config has no native env-var substitution, so the scrape
# target is templated and filled in by entrypoint.sh at container start from
# the BACKEND_URL environment variable (set this in Render's dashboard to
# the backend service's public hostname, e.g. wind-power-backend.onrender.com).
FROM prom/prometheus:v2.55.1

COPY prometheus.yml.template /etc/prometheus/prometheus.yml.template
COPY entrypoint.sh /entrypoint.sh

EXPOSE 9090
ENTRYPOINT ["/entrypoint.sh"]
