#!/bin/bash
# grid2op task entrypoint: start the SIM-API backend as a pure sim server, then
# keep the container alive (Harbor's healthcheck gates readiness; the agent and
# verifier docker-exec in later). Episode params come from the SIM_* env vars
# (set per task via task.toml [environment.env]); defaults below are for smoke runs.
set -e

export SIM_CHRONIC="${SIM_CHRONIC:-0}"
export SIM_HORIZON="${SIM_HORIZON:-24}"
export SIM_SEED="${SIM_SEED:-0}"
export SIM_ENV="${SIM_ENV:-l2rpn_case14_sandbox}"
# Training tasks never tell the agent how long the episode runs (status/observe omit max_t).
export SIM_HIDE_HORIZON="${SIM_HIDE_HORIZON:-1}"
# `observe --changes` default thresholds (agent can override per call). On idf_2023 a calm step at
# 0.01 / 1 MW lists ~60 lines (~5.8k tokens); 0.05 / 5 MW lists ~13 (~1.7k).
export SIM_CHANGES_RHO_EPS="${SIM_CHANGES_RHO_EPS:-0.05}"
export SIM_CHANGES_MW_EPS="${SIM_CHANGES_MW_EPS:-5}"
export OPENCODE_DISABLE=1
export SIMCTL_BACKEND_PORT="${SIMCTL_BACKEND_PORT:-8731}"
export MPLBACKEND=Agg

echo "[entrypoint] episode env=$SIM_ENV chronic=$SIM_CHRONIC horizon=$SIM_HORIZON seed=$SIM_SEED port=$SIMCTL_BACKEND_PORT"

# Operator token: gates the privileged routes (/sim/reset, multi-step, /bench, ...).
# Passed ONLY to the uvicorn process (not exported, so docker-exec'd shells never see
# it) and kept in a root-only file in case the verifier needs it.
tok="$(python3 -c 'import secrets;print(secrets.token_hex(24))')"
umask 077
mkdir -p /run/gridzero
chmod 700 /run/gridzero
printf '%s' "$tok" > /run/gridzero/token
chmod 600 /run/gridzero/token

# Operator manual: append this grid's facts to /app/AGENTS.md (root-owned; once per container).
facts="/app/docs/grid_facts/$SIM_ENV.md"
if [ ! -f "$facts" ]; then
  echo "[entrypoint] FATAL: no grid facts for SIM_ENV=$SIM_ENV ($facts)" >&2
  exit 1
fi
if ! grep -q '^## Grid facts' /app/AGENTS.md; then
  printf '\n' >> /app/AGENTS.md
  cat "$facts" >> /app/AGENTS.md
fi

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
