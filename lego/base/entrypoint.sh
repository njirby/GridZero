#!/bin/bash
# grid2op task entrypoint: start the SIM-API backend as a pure sim server, then
# keep the container alive (Harbor's healthcheck gates readiness; the agent and
# verifier docker-exec in later). Episode params come from the SIM_* env vars
# (set per task via task.toml [environment.env]); defaults below are for smoke runs.
set -e

export SIM_CHRONIC="${SIM_CHRONIC:-0}"
export SIM_HORIZON="${SIM_HORIZON:-24}"
export SIM_SEED="${SIM_SEED:-0}"
export OPENCODE_DISABLE=1
export SIMCTL_BACKEND_PORT="${SIMCTL_BACKEND_PORT:-8731}"
export MPLBACKEND=Agg

echo "[entrypoint] episode chronic=$SIM_CHRONIC horizon=$SIM_HORIZON seed=$SIM_SEED port=$SIMCTL_BACKEND_PORT"

# Operator token: gates the privileged routes (/sim/reset, multi-step, /bench, ...).
# Passed ONLY to the uvicorn process (not exported, so docker-exec'd shells never see
# it) and kept in a root-only file in case the verifier needs it.
tok="$(python3 -c 'import secrets;print(secrets.token_hex(24))')"
umask 077
mkdir -p /run/gridzero
chmod 700 /run/gridzero
printf '%s' "$tok" > /run/gridzero/token
chmod 600 /run/gridzero/token

mkdir -p /logs
cd /app
# Loopback only: with network_mode=public, sibling trials must not reach this backend.
SIM_API_TOKEN="$tok" nohup python -m uvicorn backend.app.main:app \
    --host 127.0.0.1 --port "$SIMCTL_BACKEND_PORT" \
    > /logs/backend.log 2>&1 &
umask 022
unset tok

# Short readiness poll (the Harbor [environment.healthcheck] is the real gate).
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$SIMCTL_BACKEND_PORT/sim/status" >/dev/null 2>&1; then
    echo "[entrypoint] backend ready"
    break
  fi
  sleep 1
done

exec sleep infinity
