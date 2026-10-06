# gridzero-rl-base — shared runtime for the grid2op Harbor task.
# Build context = the GridZero repo root:
#   docker build -f lego/base/Dockerfile -t gridzero-rl-base:v1 .
#
# Contains: python 3.12 + grid2op stack + the SIM-API backend + simctl + agent docs.
# Each per-task image is `FROM gridzero-rl-base:v1` + one COPY of episode.json.

FROM python:3.12-slim

# System deps: curl (healthcheck) + matplotlib(Agg) runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the working GridZero venv (backend/app imports resolved against these).
RUN pip install --no-cache-dir \
        "grid2op==1.12.4" "pandapower==3.4.0" \
        "fastapi==0.136.1" "uvicorn==0.46.0" "starlette==0.52.1" \
        "pydantic==2.12.3" "pydantic-settings==2.14.0" \
        "numpy==2.3.5" "pandas==2.3.3" "scipy==1.16.3" "matplotlib==3.10.9" \
        "requests==2.33.1" "psutil==7.2.2" "httpx==0.28.1" \
        "sse-starlette==3.4.2" "prometheus-fastapi-instrumentator==7.1.0"

# Backend package + CLI + agent-facing docs (the operator manual is AGENTS.md,
# which opencode auto-loads from its cwd=/app).
WORKDIR /app
COPY backend/ /app/backend/
COPY cli/simctl /usr/local/bin/simctl
RUN chmod +x /usr/local/bin/simctl
COPY AGENTS.md /app/AGENTS.md
COPY docs/ /app/docs/
COPY recipes/ /app/recipes/

# grid2op env data (baked in so containers skip the 294MB first-use download).
# grid2op looks in ~/data_grid2op (image user is root -> /root/data_grid2op).
COPY data_grid2op/ /root/data_grid2op/

# The SIM-API backend is a pure sim server here (opencode is launched by the
# Harbor agent, not the backend).
ENV OPENCODE_DISABLE=1 \
    RENDER_DISABLED=1 \
    SIMCTL_BACKEND_PORT=8731 \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

COPY lego/base/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8731
ENTRYPOINT ["/entrypoint.sh"]
