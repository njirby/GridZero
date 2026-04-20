"""Evaluation script: run rollouts with a trained checkpoint and report metrics."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Load a trained checkpoint and evaluate on grid2op scenarios."""
    import asyncio
    import torch
    from gridzero.encoder import GraphObsEncoder, FlatObsEncoder
    from gridzero.env.wrapper import make_env
    from gridzero.inference.constrained_gen import build_guided_json_params
    from gridzero.inference.vllm_client import VLLMRolloutClient
    from gridzero.policy.tokenizer_utils import build_prompt
    from gridzero.training.buffer import RolloutBuffer
    from gridzero.training.rollout import RolloutCollector

    torch.manual_seed(cfg.seed)

    gym_env, obs_parser = make_env(cfg)

    if cfg.encoder.type == "graph":
        encoder = GraphObsEncoder(cfg.encoder, node_feat_dim=obs_parser.node_feature_dim)
    else:
        encoder = FlatObsEncoder(cfg.encoder, flat_dim=obs_parser.flat_dim)

    vllm_client = VLLMRolloutClient(cfg)
    json_schema = build_guided_json_params(gym_env.action_space)
    prompt = build_prompt()

    collector = RolloutCollector(
        encoder=encoder,
        vllm_client=vllm_client,
        json_schema=json_schema,
        cfg=cfg,
    )
    buffer = RolloutBuffer(max_size=10_000)

    n_eval = cfg.get("n_eval_episodes", 10)
    episodes = asyncio.run(
        collector.collect_batch(
            env_factory=lambda: make_env(cfg)[0],
            action_space=gym_env.action_space,
            prompt=prompt,
            buffer=buffer,
            n_episodes=n_eval,
        )
    )

    survival_rate = sum(e.survived for e in episodes) / len(episodes)
    mean_steps = sum(e.total_steps for e in episodes) / len(episodes)
    mean_reward = buffer.as_tensors()["rewards"].mean().item()

    print(f"Eval over {n_eval} episodes:")
    print(f"  survival rate : {survival_rate:.2%}")
    print(f"  mean steps    : {mean_steps:.1f} / {cfg.env.max_steps}")
    print(f"  mean reward   : {mean_reward:.4f}")


if __name__ == "__main__":
    main()
