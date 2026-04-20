"""GSPO training entrypoint."""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build all components from config and run the GSPO training loop."""
    import torch
    from gridzero.encoder import GraphObsEncoder, FlatObsEncoder
    from gridzero.env.wrapper import make_env
    from gridzero.inference.constrained_gen import build_guided_json_params
    from gridzero.inference.vllm_client import VLLMRolloutClient
    from gridzero.policy.model import GridZeroPolicy
    from gridzero.training.gspo import GSPOTrainer

    torch.manual_seed(cfg.seed)

    # Environment factory (new env per episode for parallel rollouts)
    gym_env, obs_parser = make_env(cfg)
    env_factory = lambda: make_env(cfg)[0]  # noqa: E731

    # Encoder
    if cfg.encoder.type == "graph":
        encoder = GraphObsEncoder(cfg.encoder, node_feat_dim=obs_parser.node_feature_dim)
    else:
        encoder = FlatObsEncoder(cfg.encoder, flat_dim=obs_parser.flat_dim)

    # Policy (randomly initialized)
    policy = GridZeroPolicy(cfg.policy)
    print(f"Policy parameters: {policy.num_parameters:,}")

    # vllm client (server must be running — see scripts/serve_vllm.sh)
    vllm_client = VLLMRolloutClient(cfg)

    trainer = GSPOTrainer(
        policy=policy,
        encoder=encoder,
        vllm_client=vllm_client,
        env_factory=env_factory,
        cfg=cfg,
    )
    trainer.run(n_updates=cfg.training.n_updates)


if __name__ == "__main__":
    main()
