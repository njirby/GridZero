#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GridZero Experiment Runner                                            ║
# ║                                                                        ║
# ║  Edit parameters below, then:  ./scripts/run_experiment.sh             ║
# ║  All params are logged to wandb so you can compare across runs.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
set -euo pipefail

# ─── Experiment Identity ─────────────────────────────────────────────────
EXPERIMENT=""                        # optional prefix (e.g. "high-lr", "long-seq")
WANDB_PROJECT="gridzero"             # wandb project
WANDB_TAGS="baseline,case14"         # comma-separated tags for filtering
SEED=42
GPU=0                                # CUDA device index

# ─── Model ───────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-0.6B"             # HF model name or local path

# ─── Observation Encoder ─────────────────────────────────────────────────
ENCODER_TYPE="flat"                  # "flat" or "graph"
ENCODER_SEQ_LEN=16                   # number of embedding tokens injected into prompt
ENCODER_N_LAYERS=1                   # encoder depth

# ─── Environment ─────────────────────────────────────────────────────────
ENV_NAME="l2rpn_case14_sandbox"      # grid2op environment name

# ─── GRPO Loss ───────────────────────────────────────────────────────────
EPSILON=0.2                          # PPO-style clip range
BETA=0.0                             # KL penalty coefficient (0 = no KL)

# ─── Optimization ────────────────────────────────────────────────────────
LEARNING_RATE=1e-4                   # peak LR (cosine decay to 0)
MAX_GRAD_NORM=1.0                    # gradient clipping
BATCH_SIZE=1                         # per-device train batch size
GRAD_ACCUM=4                         # gradient accumulation steps
                                     # effective batch = BATCH_SIZE * GRAD_ACCUM * NUM_GENERATIONS

# ─── Generation / Rollouts ───────────────────────────────────────────────
NUM_GENERATIONS=8                    # completions per prompt (GRPO group size)
MAX_COMPLETION_LENGTH=128            # max tokens per completion
MAX_TOOL_ITERATIONS=1                # tool call rounds per completion

# ─── Schedule ────────────────────────────────────────────────────────────
MAX_STEPS=200                       # total training steps
WARMUP_STEPS=0                       # number of LR warmup steps
SAVE_STEPS=50                       # checkpoint every N steps
LOGGING_STEPS=1                      # log metrics every N steps

# ─── vLLM ────────────────────────────────────────────────────────────────
VLLM_GPU_MEMORY=0.3                  # fraction of GPU memory for vLLM
VLLM_MAX_MODEL_LEN=2048             # max sequence length (prompt + completion)

# ─── Memory ──────────────────────────────────────────────────────────────
GRADIENT_CHECKPOINTING=true          # trade compute for memory

# ══════════════════════════════════════════════════════════════════════════
#  Nothing below here needs editing for normal experiments.
# ══════════════════════════════════════════════════════════════════════════

# Auto-generate unique run name: {experiment}-{encoder}-{env_short}-lr{lr}-g{gen}-{timestamp}
ENV_SHORT="${ENV_NAME##*_}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
if [ -n "${EXPERIMENT}" ]; then
    RUN_NAME="${EXPERIMENT}-${ENCODER_TYPE}-${ENV_SHORT}-lr${LEARNING_RATE}-g${NUM_GENERATIONS}-${TIMESTAMP}"
else
    RUN_NAME="${ENCODER_TYPE}-${ENV_SHORT}-lr${LEARNING_RATE}-g${NUM_GENERATIONS}-${TIMESTAMP}"
fi
OUTPUT_DIR="outputs/${RUN_NAME}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_TAGS="${WANDB_TAGS}"

echo "Run: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"

exec python scripts/train_embeds.py \
    seed="${SEED}" \
    output_dir="${OUTPUT_DIR}" \
    policy.model_name="${MODEL}" \
    encoder.type="${ENCODER_TYPE}" \
    encoder.seq_len="${ENCODER_SEQ_LEN}" \
    encoder.n_layers="${ENCODER_N_LAYERS}" \
    env.env_name="${ENV_NAME}" \
    training.epsilon="${EPSILON}" \
    training.beta="${BETA}" \
    training.learning_rate="${LEARNING_RATE}" \
    training.max_grad_norm="${MAX_GRAD_NORM}" \
    training.per_device_train_batch_size="${BATCH_SIZE}" \
    training.gradient_accumulation_steps="${GRAD_ACCUM}" \
    training.num_generations="${NUM_GENERATIONS}" \
    training.max_completion_length="${MAX_COMPLETION_LENGTH}" \
    training.max_tool_calling_iterations="${MAX_TOOL_ITERATIONS}" \
    training.max_steps="${MAX_STEPS}" \
    training.save_steps="${SAVE_STEPS}" \
    training.logging_steps="${LOGGING_STEPS}" \
    training.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    training.vllm_gpu_memory_utilization="${VLLM_GPU_MEMORY}" \
    training.vllm_max_model_length="${VLLM_MAX_MODEL_LEN}" \
    training.report_to=wandb \
    training.run_name="${RUN_NAME}" \
    training.warmup_steps="${WARMUP_STEPS}"
