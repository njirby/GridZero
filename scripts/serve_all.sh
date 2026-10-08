#!/usr/bin/env bash
# Boot the real stack: backend (sim + opencode driver) + web dev server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8731}"
cd "$ROOT"

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT

echo "[serve-all] backend on :$PORT (log: runs/backend-$PORT.log)"
mkdir -p runs
./.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port "$PORT" \
  > "runs/backend-$PORT.log" 2>&1 &

# wait for backend
for i in $(seq 1 60); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/sim/status" >/dev/null 2>&1; then break; fi
  sleep 1
done
echo "[serve-all] backend up"

echo "[serve-all] web on :5173 (log: runs/web.log)"
( cd web && npm run dev > "$ROOT/runs/web.log" 2>&1 ) &

echo "[serve-all] ready -> http://127.0.0.1:5173  (press Ctrl-C to stop)"
wait
