#!/bin/bash
# grid2op task entrypoint: start the SIM-API backend as a pure sim server, then
# keep the container alive (Harbor's healthcheck gates readiness; the agent and
# verifier docker-exec in later). Episode params resolve: env > /app/episode.json > default.
set -e

CFG_CHRONIC=""; CFG_HORIZON=""; CFG_SEED=""
if [ -f /app/episode.json ]; then
  eval "$(python3 - <<'PY'
import json
d = json.load(open("/app/episode.json"))
print(f"CFG_CHRONIC={d.get('chronic','')}")
print(f"CFG_HORIZON={d.get('horizon','')}")
print(f"CFG_SEED={d.get('seed','')}")
PY
)"
fi

export SIM_CHRONIC="${SIM_CHRONIC:-${CFG_CHRONIC:-0}}"
export SIM_HORIZON="${SIM_HORIZON:-${CFG_HORIZON:-24}}"
export SIM_SEED="${SIM_SEED:-${CFG_SEED:-0}}"
export OPENCODE_DISABLE=1
export SIMCTL_BACKEND_PORT="${SIMCTL_BACKEND_PORT:-8731}"
export MPLBACKEND=Agg

echo "[entrypoint] episode chronic=$SIM_CHRONIC horizon=$SIM_HORIZON seed=$SIM_SEED port=$SIMCTL_BACKEND_PORT"

mkdir -p /logs
cd /app
nohup python -m uvicorn backend.app.main:app \
    --host 0.0.0.0 --port "$SIMCTL_BACKEND_PORT" \
    > /logs/backend.log 2>&1 &

# Short readiness poll (the Harbor [environment.healthcheck] is the real gate).
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$SIMCTL_BACKEND_PORT/sim/status" >/dev/null 2>&1; then
    echo "[entrypoint] backend ready"
    break
  fi
  sleep 1
done

exec sleep infinity
