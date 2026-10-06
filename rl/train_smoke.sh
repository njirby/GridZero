#!/bin/bash
# T2 — 2-GPU colocated GRPO smoke (SYNC first): LoRA r=32, SDPA (no flash-attn
# wheel for torch 2.13 on 3090s), no packing/dynamic-batch (those force FA2).
# Cards 0-1 must be free (stop the bench 4B server first). 1 prompt x 2 samples
# x 1 episode = 1 training step.
#
#   bash rl/train_smoke.sh          # sync smoke
#   ASYNC=1 bash rl/train_smoke.sh  # async + partial rollout
set -e
OZH=/home/nate/Documents/openrlhf
GZ=/home/nate/Documents/GridZero
MODEL=/home/nate/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
SAVE=${GZ}/rl/ckpt-smoke

cd "$OZH"
export CUDA_VISIBLE_DEVICES=0,1
export GRZ_RL_MODEL_NAME=qwen3.5-4b
export OPENCODE_PROVIDER=vllm4b
export TOKENIZERS_PARALLELISM=false
# system nvcc is 12.0; flashinfer's JIT needs >=12.4 (--compress-mode=size).
# Use vLLM's native sampler instead (T2 also gets the native path via
# processed_logprobs mode; this keeps T0/T1-style direct engines working too).
export VLLM_USE_FLASHINFER_SAMPLER=0

ASYNC_ARGS=()
if [ "${ASYNC:-0}" = "1" ]; then
  ASYNC_ARGS=(--train.async_enable --train.partial_rollout_enable --train.async_queue_size 1)
fi

.venv/bin/python -m openrlhf.cli.train_ppo_ray \
  --actor.model_name_or_path "$MODEL" \
  --ref.model_name_or_path "$MODEL" \
  --ckpt.output_dir "$SAVE" \
  --ckpt.path "${SAVE}/ckpt" \
  --ckpt.save_hf \
  --ckpt.save_steps 1 \
  --ckpt.max_num 2 \
  --train.agent_func_path "${GZ}/rl/gridzero_agent.py" \
  --data.prompt_dataset "${GZ}/rl/prompts_smoke.jsonl" \
  --data.input_key prompt \
  --data.max_len 49152 \
  --data.max_samples 8 \
  --data.apply_chat_template \
  --rollout.max_new_tokens 4096 \
  --rollout.batch_size 1 \
  --rollout.n_samples_per_prompt 2 \
  --rollout.micro_batch_size 1 \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --train.batch_size 2 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes 1 \
  "${ASYNC_ARGS[@]}" \
  --algo.advantage.estimator group_norm \
  --algo.advantage.is_correction_level token \
  --algo.advantage.is_correction_mode mask \
  --algo.dynamic_filtering_enable \
  --algo.dynamic_filtering_range 0.0 1.0 \
  --algo.kl.use_loss \
  --algo.kl.estimator k3 \
  --algo.kl.init_coef 1e-3 \
  --actor.adam.lr 1e-5 \
  --actor.entropy_coef 0.0 \
  --ds.lora.rank 32 \
  --ds.lora.alpha 32 \
  --ds.zero_stage 3 \
  --ds.param_dtype bf16 \
  --ds.attn_implementation sdpa \
  --actor.gradient_checkpointing_enable \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node 2 \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node 2 \
  --vllm.num_engines 2 \
  --vllm.tensor_parallel_size 1 \
  --vllm.gpu_memory_utilization 0.7 \
  --vllm.enforce_eager \
  --vllm.sync_backend nccl \
  --train.colocate_all \
  --ds.enable_sleep \
  --logger.tensorboard_dir "${SAVE}/runs" \
  --logger.logging_steps 1 \
  --eval.steps -1
