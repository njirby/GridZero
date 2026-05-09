"""GRPO training with observation embedding injection.

Usage:
    python scripts/train_embeds.py
    python scripts/train_embeds.py training.max_steps=100
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from gridzero.training.embedding_trainer import EmbeddingGRPOTrainer
from gridzero.training.gspo import build_dataset, build_grpo_config
from gridzero.training.reward import grid_reward


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset = build_dataset(cfg)
    config = build_grpo_config(cfg)
    config.beta = 0.0

    trainer = EmbeddingGRPOTrainer(
        model=cfg.policy.model_name,
        args=config,
        train_dataset=dataset,
        encoder_cfg=cfg.encoder,
        reward_funcs=grid_reward,
    )
    trainer.train()


if __name__ == "__main__":
    main()
