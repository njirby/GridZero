#!/bin/bash
# grid2op verifier: read the cumulative episode reward the backend accumulated
# and emit it as a CONTINUOUS float to /logs/verifier/reward.txt.
#
# The backend (SIM-API) runs in this same container (started by the entrypoint).
# If it is unreachable, the episode produced no score -> 0.0.
set -uo pipefail
mkdir -p /logs/verifier

reward=$(python3 - <<'PY'
import json, urllib.request
try:
    d = json.load(urllib.request.urlopen("http://127.0.0.1:8731/sim/status", timeout=5))
    print(float(d["data"]["cum_reward"]))
except Exception:
    print(0.0)
PY
)

echo "$reward" > /logs/verifier/reward.txt
echo "grid2op cumulative reward: $reward"

# Persist a small summary for inspection alongside the reward.
if command -v curl >/dev/null 2>&1; then
  curl -sf "http://127.0.0.1:8731/sim/status" > /logs/verifier/final_status.json 2>/dev/null || true
fi

exit 0
