#!/usr/bin/env bash
set -euo pipefail

# Prove that GRPO training produces a learning signal on l2rpn_case14_sandbox.
#
# Runs: baseline eval → 1000-step GRPO training → checkpoint evals → learning curve.
# Exits 0 if final reward > baseline, 1 otherwise.
#
# Requirements: CUDA GPU, ~16 GB VRAM (Qwen3-0.6B + vLLM colocate).
# Runtime: ~2-4 hours depending on GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

OUTPUT_DIR="outputs/prove_learning"

echo "============================================"
echo "  GridZero: Prove Learning"
echo "  Model:  Qwen/Qwen3-0.6B"
echo "  Env:    l2rpn_case14_sandbox"
echo "  Steps:  1000"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

# Clean previous run
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing previous run at $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

python scripts/validate_learning.py \
    output_dir="$OUTPUT_DIR" \
    policy=qwen3_0_6b \
    validation.max_steps=1000 \
    validation.dataset_size=256 \
    validation.save_steps=200 \
    validation.n_eval_episodes=10 \
    validation.model_name="Qwen/Qwen3-0.6B"

echo ""
echo "Results saved to: $OUTPUT_DIR/validation/validation_results.json"
echo ""
echo "To view training logs:"
echo "  tensorboard --logdir $OUTPUT_DIR/validation"
