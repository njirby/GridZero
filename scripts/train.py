"""GSPO training entrypoint — delegates to ms-swift's GRPOTrainer.

Usage:
    python scripts/train.py                        # use configs/config.yaml defaults
    python scripts/train.py training.n_updates=500

Or invoke ms-swift directly (equivalent, more flags available):
    swift rlhf \\
        --rlhf_type grpo \\
        --importance_sampling_level sequence \\
        --model Qwen/Qwen3-4B \\
        --external_plugins gridzero/rewards/orm_plugin.py \\
        --reward_funcs grid_composite \\
        --num_generations 8 \\
        --epsilon 3e-4 \\
        --epsilon_high 4e-4 \\
        --steps_per_generation 4 \\
        --dataset_path outputs/rollout_dataset.jsonl
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build a rollout dataset from grid2op and launch ms-swift GSPO training."""
    import os
    import torch
    from gridzero.env.wrapper import make_env
    from gridzero.env.serialization import obs_to_dataset_row
    from gridzero.encoder import GraphObsEncoder, FlatObsEncoder

    torch.manual_seed(cfg.seed)

    # --- Step 1: collect a rollout dataset of (obs, prompt) rows ---
    dataset_path = os.path.join(cfg.output_dir, "rollout_dataset.jsonl")
    os.makedirs(cfg.output_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        _collect_rollout_dataset(cfg, dataset_path)

    # --- Step 2: launch ms-swift GSPO training ---
    _launch_swift_gspo(cfg, dataset_path)


def _collect_rollout_dataset(cfg: DictConfig, out_path: str, n_episodes: int = 100) -> None:
    """Roll out a random policy to collect starting observations for GSPO prompts."""
    import json
    from gridzero.env.wrapper import make_env
    from gridzero.env.serialization import obs_to_dataset_row

    print(f"Collecting rollout dataset → {out_path}")
    gym_env, obs_parser = make_env(cfg)
    rows = []

    for ep in range(n_episodes):
        obs, _ = gym_env.reset()
        done = False
        step = 0
        while not done and step < cfg.env.max_steps:
            rows.append(obs_to_dataset_row(obs, env_id=ep))
            # Random action to advance to the next state
            action = gym_env.action_space.sample()
            obs, _, done, _, _ = gym_env.step(action)
            step += 1

    gym_env.close()

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Collected {len(rows)} observations across {n_episodes} episodes.")


def _launch_swift_gspo(cfg: DictConfig, dataset_path: str) -> None:
    """Launch ms-swift GRPOTrainer with GSPO (sequence-level importance sampling)."""
    from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments  # type: ignore[import]
    from swift.plugin import orms  # type: ignore[import]

    # Register our ORM — importing the module triggers orms["grid_composite"] = ...
    import gridzero.rewards.orm_plugin  # noqa: F401

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        rlhf_type="grpo",
        importance_sampling_level="sequence",   # GSPO
        num_generations=cfg.training.n_completions_per_state,
        epsilon=cfg.training.clip_eps,
        epsilon_high=cfg.training.get("clip_eps_high", cfg.training.clip_eps * 1.33),
        steps_per_generation=cfg.training.get("steps_per_generation", 4),
        beta=cfg.training.kl_coeff,
        learning_rate=cfg.training.lr,
        max_grad_norm=cfg.training.grad_clip,
        reward_funcs=["grid_composite"],
        # TODO: add external_plugins path when running via swift CLI
    )

    # TODO: load model, tokenizer, dataset and call trainer.train()
    # This stub maps cfg → swift args; full wiring requires dataset formatting
    # to match ms-swift's expected schema (see docs/GRPO.md).
    print("ms-swift GSPO args configured:")
    print(f"  importance_sampling_level : sequence (GSPO)")
    print(f"  num_generations           : {training_args.num_generations}")
    print(f"  epsilon                   : {training_args.epsilon}")
    print(f"  beta (KL)                 : {training_args.beta}")
    print(f"  dataset                   : {dataset_path}")
    print()
    print("To launch directly with the swift CLI:")
    print(
        f"  swift rlhf \\\n"
        f"    --rlhf_type grpo \\\n"
        f"    --importance_sampling_level sequence \\\n"
        f"    --model {cfg.policy.model_name} \\\n"
        f"    --external_plugins gridzero/rewards/orm_plugin.py \\\n"
        f"    --reward_funcs grid_composite \\\n"
        f"    --num_generations {cfg.training.n_completions_per_state} \\\n"
        f"    --dataset_path {dataset_path}"
    )


if __name__ == "__main__":
    main()
