"""GRPO training with observation embedding injection.

Usage:
    python scripts/train_embeds.py
    python scripts/train_embeds.py training.max_steps=100
    ./scripts/run_experiment.sh  # recommended — exposes all knobs
"""
from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from gridzero.training.embedding_trainer import EmbeddingGRPOTrainer
from gridzero.training.gspo import build_dataset, build_grpo_config
from gridzero.training.reward import grid_reward


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.training.get("report_to") == "wandb":
        import wandb
        tags = [t.strip() for t in os.environ.get("WANDB_TAGS", "").split(",") if t.strip()]
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gridzero"),
            name=cfg.training.get("run_name"),
            tags=tags or None,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_code=True,
        )

    dataset = build_dataset(cfg)
    config = build_grpo_config(cfg)
    config.beta = 0.0

    trainer = EmbeddingGRPOTrainer(
        model=cfg.policy.model_name,
        args=config,
        train_dataset=dataset,
        encoder_cfg=cfg.encoder,
        env_name=cfg.env.env_name,
        random_init=bool(cfg.policy.get("random_init", False)),
        reward_funcs=grid_reward,
    )
    trainer.train()

    if cfg.training.get("report_to") == "wandb":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
