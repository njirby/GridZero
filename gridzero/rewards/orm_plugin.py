"""ms-swift ORM (reward function) plugin for GridZero.

Register this file with ms-swift via --external_plugins:

    swift rlhf \
        --rlhf_type grpo \
        --external_plugins gridzero/rewards/orm_plugin.py \
        --reward_funcs grid_composite \
        ...

The ORM receives the model's completions (tool call JSON strings) and the
dataset row fields (obs_flat, n_lines, etc.) as kwargs. It uses
obs.simulate(action) — grid2op's built-in non-mutating lookahead — to score
each completion without advancing the environment state.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from swift.plugin import ORM, orms  # type: ignore[import]

from gridzero.env.actions import parse_tool_call
from gridzero.rewards.grid_rewards import composite_reward


class GridCompositeORM(ORM):
    """Reward function that scores tool calls via grid2op's obs.simulate().

    obs.simulate(action) performs a one-step lookahead from the current
    observation without mutating the environment — all G completions for the
    same grid state are scored independently and correctly from the same
    starting point.

    One environment is created per ORM instance (one per ms-swift worker process).
    """

    INVALID_ACTION_PENALTY = -0.5

    def __init__(self) -> None:
        self._env = None
        self._action_space = None
        self._obs_space = None

    def _get_env(self, env_name: str = "l2rpn_case14_sandbox"):
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
            self._obs_space = self._env.observation_space
        return self._env, self._action_space, self._obs_space

    def __call__(
        self,
        completions: list[str],
        obs_flat: list[float] | None = None,
        env_name: str = "l2rpn_case14_sandbox",
        reward_weights: dict | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """Score a batch of completions using obs.simulate(action).

        Args:
            completions: JSON tool call strings from the policy (one per G sample).
            obs_flat: Serialized flat observation vector from obs_to_dataset_row().
                      All completions in a GSPO group share the same obs_flat.
            env_name: grid2op environment name.
            reward_weights: Optional composite_reward weight overrides.

        Returns:
            List of scalar rewards, one per completion.
        """
        env, action_space, obs_space = self._get_env(env_name)

        # Reconstruct the Observation object from the serialized flat vector.
        # obs_space.from_vect() is grid2op's standard deserialization path.
        obs = obs_space.from_vect(np.array(obs_flat, dtype=np.float32))

        rewards: list[float] = []
        for completion in completions:
            try:
                action = parse_tool_call(completion, action_space)
            except Exception:
                rewards.append(self.INVALID_ACTION_PENALTY)
                continue

            try:
                # simulate() does a non-mutating one-step lookahead.
                sim_obs, sim_reward, sim_done, sim_info = obs.simulate(action)
                r = composite_reward(sim_obs, sim_done, weights=reward_weights)
            except Exception:
                r = self.INVALID_ACTION_PENALTY

            rewards.append(r)

        return rewards


orms["grid_composite"] = GridCompositeORM
