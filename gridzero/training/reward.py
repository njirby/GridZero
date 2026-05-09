"""Reward function for TRL GRPOTrainer."""
from __future__ import annotations


def grid_reward(environments, **kwargs) -> list[float]:
    """Read accumulated reward from each GridEnv instance."""
    return [env.reward for env in environments]
