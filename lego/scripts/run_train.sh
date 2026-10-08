#!/usr/bin/env bash
# Launch the GridZero GRPO+LoRA run. Recreated 2026-10-07 after /tmp/opencode was lost on reboot.
# Usage: bash run_train.sh [config.env]   (logs -> $LOG_DIR/grz-train.log)
set -uo pipefail
CONFIG="${1:-/home/nate/Documents/GridZero/lego/configs/gridzero_2b_lora.env}"
LEGO=/home/nate/Documents/lego-rl
# Ray's memory monitor kills workers at 95% RAM before the kernel ever swaps — disable it
# so swap is usable. Must reach the raylet env, so ray must not already be running.
export RAY_memory_monitor_refresh_ms=0
# Cards 2-3 are reserved — training uses only 0-1.
export CUDA_VISIBLE_DEVICES=0,1
if pgrep -f "[D]ocuments/lego-rl/.venv.*(raylet|gcs_server)" >/dev/null; then
    echo "[FATAL] stale lego-rl ray is running; clean up first (see debug-journal.md)" >&2; exit 1
fi
docker image inspect gridzero-rl-base:v2 >/dev/null 2>&1 || HOME=/home/nate bash /home/nate/Documents/GridZero/lego/base/build.sh gridzero-rl-base:v2
cd "$LEGO" && exec bash scripts/train/train.sh "$CONFIG"
