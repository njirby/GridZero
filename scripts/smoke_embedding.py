"""Longer embedding-based GRPO smoke test to verify meaningful metrics.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_embedding.py
"""
from __future__ import annotations

import torch
from datasets import Dataset
from omegaconf import OmegaConf
from trl import GRPOConfig

from gridzero.training.embedding_trainer import EmbeddingGRPOTrainer
from gridzero.training.gspo import get_default_structural_tag
from gridzero.training.reward import grid_reward


def main():
    encoder_cfg = OmegaConf.create({
        "type": "flat",
        "d_model": 1024,
        "seq_len": 16,
        "n_layers": 1,
    })

    dataset = Dataset.from_dict({
        "prompt": [[] for _ in range(32)],
        "chronics_id": [i % 100 for i in range(32)],
    })

    config = GRPOConfig(
        output_dir="outputs/smoke_embedding",
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=64,
        max_tool_calling_iterations=1,
        loss_type="grpo",
        epsilon=0.2,
        beta=0.0,
        learning_rate=1e-4,
        max_steps=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=20,
        report_to="tensorboard",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=1024,
        chat_template_kwargs={"enable_thinking": False},
        generation_kwargs={"structured_outputs": {"structural_tag": get_default_structural_tag()}},
        seed=42,
        bf16=True,
        log_completions=True,
        gradient_checkpointing=True,
    )

    trainer = EmbeddingGRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        args=config,
        train_dataset=dataset,
        encoder_cfg=encoder_cfg,
        reward_funcs=grid_reward,
    )
    trainer.train()


if __name__ == "__main__":
    main()
