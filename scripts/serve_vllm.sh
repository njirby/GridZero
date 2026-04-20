#!/usr/bin/env bash
# Start the vllm inference server for GridZero rollout generation.
# Requires vllm installed and --enable-mm-embeds support (vllm >= 0.5).
#
# Usage:
#   bash scripts/serve_vllm.sh [model_path] [extra vllm args...]
#
# Environment variables:
#   GRIDZERO_MODEL_PATH  - path to HF model dir (required if not passed as arg)
#   GRIDZERO_VLLM_PORT   - port to bind (default 8000)

set -euo pipefail

MODEL_PATH="${1:-${GRIDZERO_MODEL_PATH:?Set GRIDZERO_MODEL_PATH or pass model path as first arg}}"
PORT="${GRIDZERO_VLLM_PORT:-8000}"
shift 1 2>/dev/null || true  # remove model_path from extra args if present

echo "Starting vllm server:"
echo "  model : $MODEL_PATH"
echo "  port  : $PORT"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --enable-mm-embeds \
    --port "$PORT" \
    --trust-remote-code \
    "$@"
