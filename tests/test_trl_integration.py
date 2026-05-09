"""Integration test: verify GRPOTrainer instantiates and runs with GridEnv."""
from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from gridzero.training.env import GridEnv
from gridzero.training.gspo import STRUCTURAL_TAG, SYSTEM_PROMPT
from gridzero.training.reward import grid_reward


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_grpo_trainer_runs_2_steps(tmp_path):
    dataset = Dataset.from_dict({
        "prompt": [
            [{"role": "system", "content": SYSTEM_PROMPT}]
            for _ in range(4)
        ],
        "chronics_id": [0, 1, 2, 3],
    })

    config = GRPOConfig(
        output_dir=str(tmp_path / "output"),
        num_generations=2,
        generation_batch_size=2,
        max_completion_length=256,
        max_tool_calling_iterations=1,
        loss_type="grpo",
        epsilon=0.2,
        beta=0.04,
        learning_rate=1e-5,
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=2,
        report_to="none",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        chat_template_kwargs={"enable_thinking": False},
        generation_kwargs={"structured_outputs": {"structural_tag": STRUCTURAL_TAG}},
        seed=42,
        bf16=True,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        reward_funcs=grid_reward,
        train_dataset=dataset,
        environment_factory=GridEnv,
        args=config,
    )
    trainer.train()

    assert (tmp_path / "output" / "checkpoint-2").exists()
