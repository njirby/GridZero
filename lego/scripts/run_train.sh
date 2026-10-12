#!/usr/bin/env bash
# Launch the GridZero GRPO+LoRA run. Recreated 2026-10-07 after /tmp/opencode was lost on reboot.
# Usage: bash run_train.sh [config.env]   (logs -> $LOG_DIR/grz-train.log)
set -uo pipefail
CONFIG="${1:-/home/nate/Documents/GridZero/lego/configs/gridzero_2b_lora.env}"
LEGO=/home/nate/Documents/lego-rl
# Ray's memory monitor kills workers at 95% RAM before the kernel ever swaps — disable it
# so swap is usable. Must reach the raylet env, so ray must not already be running.
export RAY_memory_monitor_refresh_ms=0
# GPUs come from the config's CUDA_VISIBLE_DEVICES (default 0,1: cards 2-3 normally run the
# user's own model). Read it in a clean subshell so the caller's environment can't leak in.
CVD=$(env -u CUDA_VISIBLE_DEVICES bash -c 'set -a; source "$1" >/dev/null 2>&1; echo "${CUDA_VISIBLE_DEVICES:-0,1}"' _ "$CONFIG")
export CUDA_VISIBLE_DEVICES="$CVD"
# Refuse to launch onto a GPU something else is computing on (e.g. the user's model on 2-3).
busy=""
for g in ${CVD//,/ }; do
    apps=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -n "$apps" ] && busy="$busy $g(pids:$(echo $apps | tr '\n' ' '))"
done
if [ -n "$busy" ] && [ "${ALLOW_BUSY_GPUS:-0}" != 1 ]; then
    echo "[FATAL] GPUs in use:$busy — free them or set ALLOW_BUSY_GPUS=1" >&2; exit 1
fi
echo "[run_train] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# Show every line (Ray otherwise collapses identical-looking worker lines, hiding diagnostics).
export RAY_DEDUP_LOGS=0
if pgrep -f "[D]ocuments/lego-rl/.venv.*(raylet|gcs_server)" >/dev/null; then
    echo "[FATAL] stale lego-rl ray is running; clean up first (see debug-journal.md)" >&2; exit 1
fi
docker image inspect gridzero-rl-base:v2 >/dev/null 2>&1 || HOME=/home/nate bash /home/nate/Documents/GridZero/lego/base/build.sh gridzero-rl-base:v2
cd "$LEGO" && exec bash scripts/train/train.sh "$CONFIG"
