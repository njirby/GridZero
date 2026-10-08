#!/bin/bash
# grid2op verifier: read the cumulative episode reward the backend accumulated
# and emit it as a CONTINUOUS float to /logs/verifier/reward.txt.
#
# The backend (SIM-API) runs in this same container (started by the entrypoint).
# If it is unreachable the episode produced no score: that is an INFRA failure, not
# a real 0.0, so reward.txt is deliberately NOT written and the script exits 1
# (/logs/verifier/INFRA_FAILURE says why). Consequences, verified in lego-rl:
#   - Harbor verifier.py:148 raises RewardFileNotFoundError -> trial.run() returns no
#     verifier_result -> builtin_swe_agent_loop.py:~830 "no verifier result, retrying"
#     re-runs the WHOLE trial (agent included).
#   - Retries are BOUNDED: max_retries = HARBOR_MAX_RETRIES (default 2, see
#     agent_loop_config_oc_docker.yaml:14), so a task whose backend never starts costs
#     at most 2 attempts, never loops forever.
#   - CAVEAT: when all attempts fail, _run_harbor_trial still returns reward=0.0 with
#     reason agent_completed (builtin_swe_agent_loop.py:~667), so a persistently broken
#     task is trained on as a real 0.0, NOT dropped (only timeout / env_setup_failed are
#     in trajectory_filter's DEFAULT_DROP_REASONS). A dead backend usually also means a
#     garbage trajectory; fixing that needs a lego-rl change (map "no verifier result
#     after all retries" to env_setup_failed).
set -uo pipefail
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/INFRA_FAILURE

reward=$(python3 - <<'PY'
import json, urllib.request
try:
    d = json.load(urllib.request.urlopen("http://127.0.0.1:8731/sim/status", timeout=5))
    print(float(d["data"]["cum_reward"]))
except Exception as e:
    print(f"ERR {type(e).__name__}: {e}")
PY
)

if [[ ! "$reward" =~ ^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
  echo "verifier infra failure: backend status unavailable ($reward)" | tee /logs/verifier/INFRA_FAILURE >&2
  exit 1
fi

echo "$reward" > /logs/verifier/reward.txt
echo "grid2op cumulative reward: $reward"

# Persist a small summary for inspection alongside the reward.
curl -sf "http://127.0.0.1:8731/sim/status" > /logs/verifier/final_status.json 2>/dev/null || true

exit 0
