"""Tests for the TRL reward function."""
from __future__ import annotations

from gridzero.training.reward import grid_reward


class _MockEnv:
    def __init__(self, reward: float):
        self.reward = reward


def test_reward_returns_floats():
    envs = [_MockEnv(0.5), _MockEnv(-0.3), _MockEnv(1.0)]
    rewards = grid_reward(envs)
    assert rewards == [0.5, -0.3, 1.0]


def test_reward_length_matches_envs():
    envs = [_MockEnv(r) for r in range(10)]
    rewards = grid_reward(envs)
    assert len(rewards) == 10


def test_empty_envs():
    assert grid_reward([]) == []
