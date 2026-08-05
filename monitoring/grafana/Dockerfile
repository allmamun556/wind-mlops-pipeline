# Standalone Grafana image for a direct Render deploy (separate from the
# docker-compose stack, which instead bind-mounts these directories).
#
# The Prometheus datasource URL is resolved from the PROMETHEUS_URL
# environment variable at runtime via Grafana's built-in provisioning
# env-var expansion (see provisioning/datasources/prometheus.yml) — set it
# in Render's dashboard to the Prometheus service's public URL, e.g.
# https://wind-power-prometheus.onrender.com.
FROM grafana/grafana:11.3.1

COPY provisioning /etc/grafana/provisioning
COPY dashboards /etc/grafana/dashboards

# Anonymous viewer access — matches docker-compose.yml's local default
# (portfolio-demo convenience; the user explicitly chose this over requiring
# login for the public deploy too).
ENV GF_AUTH_ANONYMOUS_ENABLED=true
ENV GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
ENV GF_AUTH_DISABLE_LOGIN_FORM=true

EXPOSE 3000
