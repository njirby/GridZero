"""GRPO training entrypoint using TRL GRPOTrainer with live grid2op environment.

Usage:
    python scripts/train.py
    python scripts/train.py training.max_steps=100
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig
from trl import GRPOTrainer

from gridzero.training.env import GridEnv
from gridzero.training.gspo import build_dataset, build_grpo_config
from gridzero.training.reward import grid_reward


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset = build_dataset(cfg)
    config = build_grpo_config(cfg)

    trainer = GRPOTrainer(
        model=cfg.policy.model_name,
        reward_funcs=grid_reward,
        train_dataset=dataset,
        environment_factory=GridEnv,
        args=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
