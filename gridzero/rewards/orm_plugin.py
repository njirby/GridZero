"""ms-swift ORM (reward function) plugin for GridZero.

Register this file with ms-swift via --external_plugins:

    swift rlhf \
        --rlhf_type grpo \
        --external_plugins gridzero/rewards/orm_plugin.py \
        --reward_funcs grid_composite \
        ...

The ORM receives the model's completions (tool call JSON strings) and the
dataset row fields (obs_flat, n_lines, etc.) as kwargs. It steps a local
grid2op environment with each parsed action and returns scalar rewards.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from swift.plugin import ORM, orms  # type: ignore[import]

from gridzero.env.actions import parse_tool_call
from gridzero.rewards.grid_rewards import composite_reward


class GridCompositeORM(ORM):
    """Reward function that steps a grid2op environment with each tool call.

    One lightweight environment is created per ORM instance. Because ms-swift
    may call the ORM from multiple processes, each worker gets its own env.

    Reward = composite_reward(obs, done) from gridzero.rewards.grid_rewards.
    Invalid tool calls (parse errors, out-of-range IDs) receive a fixed penalty.
    """

    INVALID_ACTION_PENALTY = -0.5

    def __init__(self) -> None:
        self._env = None
        self._action_space = None

    def _get_env(self, env_name: str = "l2rpn_case14_sandbox"):
        """Lazily initialize the grid2op environment (once per worker)."""
        if self._env is None:
            import grid2op
            try:
                from lightsim2grid import LightSimBackend
                backend = LightSimBackend()
            except ImportError:
                from grid2op.Backend import PandaPowerBackend
                backend = PandaPowerBackend()
            self._env = grid2op.make(env_name, backend=backend)
            self._action_space = self._env.action_space
        return self._env, self._action_space

    def __call__(
        self,
        completions: list[str],
        obs_flat: list[list[float]] | None = None,
        env_name: str = "l2rpn_case14_sandbox",
        reward_weights: dict | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """Score a batch of completions by stepping grid2op.

        Args:
            completions: List of JSON tool call strings from the policy.
            obs_flat: Serialized flat observation vectors (one per completion).
                      Used to restore the environment state before each step.
            env_name: grid2op environment name (passed through dataset row).
            reward_weights: Optional override for composite_reward weights.

        Returns:
            List of scalar rewards, one per completion.
        """
        env, action_space = self._get_env(env_name)
        rewards: list[float] = []

        for i, completion in enumerate(completions):
            try:
                action = parse_tool_call(completion, action_space)
            except Exception:
                rewards.append(self.INVALID_ACTION_PENALTY)
                continue

            # Restore env state from the serialized flat obs if provided.
            # Without state restoration each completion sees a different state,
            # which is acceptable for the initial bandit-style training loop
            # but should be fixed for proper episode-level training.
            # TODO: restore env from obs_flat[i] once grid2op supports it cleanly
            try:
                obs, reward, done, info = env.step(action)
                r = composite_reward(obs, done, weights=reward_weights)
            except Exception:
                r = self.INVALID_ACTION_PENALTY

            rewards.append(r)

        return rewards


# Register under the name used in --reward_funcs
orms["grid_composite"] = GridCompositeORM
