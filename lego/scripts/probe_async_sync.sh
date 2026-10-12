#!/usr/bin/env bash
# P1 smoke for the async 2+2 run (configs/gridzero_2b_lora_async.env): 2 updates on a tiny batch,
# with the weight-sync canary on (VERL_SYNC_CANARY=1, verl/checkpoint_engine/base.py SyncCanary).
#
# Pass criteria (parsed from the run log):
#   - canary: a fixed token sequence is scored by each vLLM replica before the first sync and after
#     every sync, and by the trainer. LoRA B=0 at init, so version 0 equals the base by design;
#     after update 1 the vLLM log-probs must CHANGE vs pre-sync (mean |d| > 1e-3) and AGREE with
#     the trainer (mean |d| < 0.02, bf16 floor). Same agreement after update 2.
#   - trainer and rollout processes report disjoint GPU UUIDs ([GPU-PLACEMENT] lines);
#   - rollouts ran BuiltinSWEAgentLoop trials (no single_turn_agent);
#   - memory PSI stayed sane (the run is aborted if some avg10 > 40% for 45 s or avg60 > 30%).
#
# Usage: bash lego/scripts/probe_async_sync.sh        (log -> lego/logs/probe-async-<ts>.log)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LEGO_DIR=$(dirname "$HERE")
CONFIG="$LEGO_DIR/configs/gridzero_2b_lora_async.env"
PY=/home/nate/Documents/lego-rl/.venv/bin/python
TS=$(date +%Y%m%d-%H%M%S)
LOG="$LEGO_DIR/logs/probe-async-$TS.log"
PSI_LOG="$LEGO_DIR/logs/probe-async-$TS.psi"
TIMEOUT_S=${PROBE_TIMEOUT_S:-5400}
mkdir -p "$LEGO_DIR/logs"

# --- preconditions: GPUs 0-3 idle, no trace capture, no stale lego-rl ray ---
apps=$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
[ -z "$apps" ] || { echo "[probe] FATAL: GPUs 0-3 busy (pids: $(echo $apps))" >&2; exit 2; }
if tmux ls 2>/dev/null | grep -q '^grz-trace-'; then echo "[probe] FATAL: grz-trace-* tmux session running" >&2; exit 2; fi

# --- tiny run: 2 prompts x 4 rollouts per update, 4 prompts total -> 2 updates (+ the initial sync) ---
export EXP_NAME="probe-async-$TS"
export TRAIN_MINI_BSZ=2 N_RESP=4 TOTAL_ROLLOUT_STEPS=4
export MAX_RESP=8192 HARBOR_AGENT_MAX_ITERATIONS=${HARBOR_AGENT_MAX_ITERATIONS:-8}
export OC_CONTEXT_LIMIT=49152 OC_OUTPUT_LIMIT=8192   # must fit MAX_PROMPT + MAX_RESP
export TRAINER_VAL_BEFORE_TRAIN=False TRAINER_TEST_FREQ=-1 TRAINER_SAVE_FREQ=-1
export TRAINER_LOGGER="['console']"
export VERL_SYNC_CANARY=1

containers_before=$(docker ps -aq --filter name=oc-gridzero | sort)
echo "[probe] log: $LOG"
setsid bash "$HERE/run_train.sh" "$CONFIG" > "$LOG" 2>&1 < /dev/null &
RUN_PID=$!

cleanup() {
    echo "[probe] stopping run (pid $RUN_PID) and cleaning up"
    kill -TERM -- "-$RUN_PID" 2>/dev/null || true
    sleep 5
    /home/nate/Documents/lego-rl/.venv/bin/ray stop --force >/dev/null 2>&1 || true
    pkill -9 -f "[D]ocuments/lego-rl/.venv" 2>/dev/null || true
    for c in $(comm -13 <(echo "$containers_before") <(docker ps -aq --filter name=oc-gridzero | sort)); do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
    for n in $(docker network ls --format '{{.Name}}' | grep '^oc-gridzero' || true); do
        docker network rm "$n" >/dev/null 2>&1 || true
    done
}

# --- wait: until the post-sync canary for version 2 is logged, the run exits, PSI trips, or timeout ---
start=$(date +%s); hot=0; status=""
while :; do
    some10=$(awk '/^some/{split($2,a,"="); print a[2]}' /proc/pressure/memory)
    some60=$(awk '/^some/{split($3,a,"="); print a[2]}' /proc/pressure/memory)
    full10=$(awk '/^full/{split($2,a,"="); print a[2]}' /proc/pressure/memory)
    echo "$(date +%T) some10=$some10 some60=$some60 full10=$full10 $(free -m | awk '/Mem:/{print "used_mb="$3} /Swap:/{print "swap_mb="$3}' | tr '\n' ' ')" >> "$PSI_LOG"
    if awk -v v="$some10" 'BEGIN{exit !(v > 40)}'; then hot=$((hot + 5)); else hot=0; fi
    if [ "$hot" -ge 45 ] || awk -v v="$some60" 'BEGIN{exit !(v > 30)}'; then status="ABORT_PSI"; break; fi
    if grep -q '"src": "trainer", "tag": "post_sync", "version": 2' "$LOG" 2>/dev/null; then status="DONE"; break; fi
    if ! kill -0 "$RUN_PID" 2>/dev/null; then status="EXITED"; break; fi
    [ $(( $(date +%s) - start )) -ge "$TIMEOUT_S" ] && { status="TIMEOUT"; break; }
    sleep 5
done
echo "[probe] wait ended: $status after $(( $(date +%s) - start ))s"
# The last update's metrics are logged after its canary: let the run finish on its own (<=5 min).
if [ "$status" = DONE ]; then
    for _ in $(seq 60); do kill -0 "$RUN_PID" 2>/dev/null || break; sleep 5; done
fi
cleanup

"$PY" - "$LOG" "$PSI_LOG" "$status" <<'PY'
import json, re, sys
import numpy as np

log_path, psi_path, status = sys.argv[1:4]
text = open(log_path, errors="replace").read()
lines = text.splitlines()
ok = True

def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg)

def records(tag):
    # raw_decode: Ray sometimes glues the next line's prefix onto a long line.
    out, dec = [], json.JSONDecoder()
    for line in lines:
        i = line.find(f"[{tag}] ")
        if i >= 0:
            try:
                out.append(dec.raw_decode(line[i + len(tag) + 3:])[0])
            except json.JSONDecodeError:
                print(f"  unparsable {tag} line: {line[:120]}")
    return out

print(f"run status: {status}")

# GPU placement
place = records("GPU-PLACEMENT")
trainer_uuids = {r["uuid"] for r in place if r["role"] == "trainer"}
rollout_uuids = {r["uuid"] for r in place if r["role"].startswith("rollout")}
for r in place:
    print(f"  placement {r['role']:<13} ray_gpu_ids={r.get('ray_gpu_ids')} cvd={r.get('cuda_visible_devices')} uuid={r['uuid']}")
check(trainer_uuids and rollout_uuids and not (trainer_uuids & rollout_uuids),
      f"GPU sets disjoint: trainer={len(trainer_uuids)} rollout={len(rollout_uuids)} overlap={len(trainer_uuids & rollout_uuids)}")

# Canary
can = records("SYNC-CANARY")
def lp(src, tag, version, replica=None):
    for r in can:
        if r["src"] == src and r["tag"] == tag and r["version"] == version and (replica is None or r.get("replica") == replica):
            return np.array(r["logprobs"])
    return None
def mad(a, b):
    return float(np.mean(np.abs(a - b))) if a is not None and b is not None else float("nan")

replicas = sorted({r["replica"] for r in can if r["src"] == "vllm"})
print(f"  canary records: {len(can)}, replicas: {replicas}")
# "Agree" is judged against the measured base floor: vLLM (bf16) vs the FSDP trainer (SP2, fused
# kernels) already differ by ~0.028 mean |d| on the BASE model (probe 2026-10-12), so an absolute
# 0.02 is unattainable. After a sync, vLLM must be as close to the current trainer as the base floor
# allows, and closer to it than to the pre-sync (stale) trainer.
pre_t = lp("trainer", "pre_sync", 0)
for rep in replicas:
    pre_v = lp("vllm", "pre_sync", 0, rep)
    floor = mad(pre_v, pre_t)
    print(f"  replica {rep}: base floor vllm-vs-trainer {floor:.5f}; v0 vs pre {mad(lp('vllm', 'post_sync', 0, rep), pre_v):.5f}")
    for v in (1, 2):
        post_v, post_t = lp("vllm", "post_sync", v, rep), lp("trainer", "post_sync", v)
        change, agree, stale = mad(post_v, pre_v), mad(post_v, post_t), mad(post_v, pre_t)
        print(f"  replica {rep} v{v}: change vs pre {change:.5f}, vs trainer v{v} {agree:.5f}, "
              f"vs pre-sync trainer {stale:.5f}, trainer change {mad(post_t, pre_t):.5f}")
        check(change > 1e-3, f"replica {rep} v{v}: vLLM log-probs changed after sync (mean|d|={change:.5f} > 1e-3)")
        check(agree <= 1.25 * floor + 1e-3 and agree < stale,
              f"replica {rep} v{v}: vLLM agrees with trainer v{v} ({agree:.5f} <= 1.25 x floor {floor:.5f}, < stale {stale:.5f})")
check(len(replicas) == 2, "two vLLM replicas scored")

# Sync size / time, adapter load
for m in re.findall(r"\[LORA-SYNC\][^\n]*", text):
    print("  " + m)
for m in re.findall(r"\[SYNC-TIME\][^\n]*", text):
    print("  " + m)
loaded = re.findall(r"\[LORA-LOADED\][^\n]*", text)
for m in loaded:
    print("  " + m)
check(loaded, "vLLM loaded the LoRA adapter (add_lora)")

# Agent loop
check("single_turn_agent" not in text, "no single_turn_agent in log")
check(re.search(r"BuiltinSWE|builtin_swe_agent", text), "builtin_swe_agent loop used")

# Step metrics (console logger: 'step:N - key:val - ...')
# GSPO + bypass: the sequence ratio vs the rollout log-probs includes the vLLM/FSDP mismatch, so
# watch how much of the batch is clipped (clip 3e-4/4e-4 vs a ~5e-4 mismatch seen in sync runs).
keys = ("actor/grad_norm", "actor/pg_loss", "actor/pg_clipfrac", "actor/pg_clipfrac_lower", "actor/ppo_kl",
        "rollout_corr/kl", "rollout_corr/log_ppl_abs_diff", "training/rollout_probs_diff_mean",
        "critic/score/mean", "timing_s/step", "timing_s/param_sync", "timing_s/update_actor",
        "fully_async/trainer/idle_ratio", "fully_async/rollouter/active_time", "fully_async/rollouter/idle_ratio",
        "response_length/mean")
clipfrac = {}
for line in lines:
    m = re.search(r"step:(\d+) - ", line)
    if not m:
        continue
    found = {k: v for k, v in re.findall(r"([\w/@.]+):([-\d.eE+naif]+)", line) if k in keys or "clip" in k}
    if found:
        print(f"  step {m.group(1)}: " + " ".join(f"{k}={v}" for k, v in found.items()))
        if "actor/pg_clipfrac" in found:
            clipfrac[int(m.group(1))] = float(found["actor/pg_clipfrac"])
if clipfrac:
    first = clipfrac[min(clipfrac)]
    print(("WARN " if first > 0.3 else "INFO ") + f"GSPO pg_clipfrac at first update = {first:.3f} (>0.3 means bypass mismatch clips most sequences)")

# PSI
peak = {"some10": 0.0, "some60": 0.0, "full10": 0.0, "used_mb": 0, "swap_mb": 0}
for line in open(psi_path):
    for k, v in re.findall(r"(\w+)=([\d.]+)", line):
        if k in peak:
            peak[k] = max(peak[k], float(v))
print(f"  PSI/RAM peak: {peak}")
check(status in ("DONE",), f"run reached the version-2 canary (status={status})")
print("PROBE " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
PY
