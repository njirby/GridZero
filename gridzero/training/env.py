"""TRL-compatible grid2op environment for GRPO training.

Exposes grid control actions as tool methods that TRL's GRPOTrainer
auto-discovers via typed signatures and docstrings.
"""
from __future__ import annotations

import json

import grid2op
import numpy as np

from gridzero.env.observation import ObsParser
from gridzero.rewards.grid_rewards import composite_reward


def _format_obs_data(obs_data) -> str:
    """Format an ObsData as a compact JSON string."""
    source = obs_data.raw if obs_data.raw is not None else obs_data
    fields: dict = {}
    for key, attr in [
        ("rho", "rho"),
        ("line_status", "line_status"),
        ("load_p", "load_p"),
        ("gen_p", "gen_p"),
        ("v_or", "v_or"),
    ]:
        arr = getattr(source, attr, None)
        if arr is not None:
            if isinstance(arr, np.ndarray):
                fields[key] = [round(float(v), 4) for v in arr]
            else:
                fields[key] = arr
    fields["n_lines"] = obs_data.n_lines
    fields["n_loads"] = obs_data.n_loads
    fields["n_gens"] = obs_data.n_gens
    return json.dumps(fields, separators=(",", ":"))


class GridEnv:
    """Live grid2op environment for TRL's ``environment_factory``.

    Each public method (except ``reset``) is exposed as a tool that the LLM
    can call. TRL generates the tool schema from the method signature and
    docstring automatically.
    """

    def __init__(self, env_name: str = "l2rpn_case14_sandbox") -> None:
        try:
            from lightsim2grid import LightSimBackend
            backend = LightSimBackend()
        except ImportError:
            from grid2op.Backend import PandaPowerBackend
            backend = PandaPowerBackend()
        self._env = grid2op.make(env_name, backend=backend)
        self._obs_parser = ObsParser(self._env)
        self._obs = None
        self._done = False
        self.reward = 0.0
        self._last_obs_data = None

    @property
    def last_obs_data(self):
        """The most recent parsed observation (ObsData), or None before reset."""
        return self._last_obs_data

    def reset(self, chronics_id: int = 0, **kwargs) -> str:
        """Initialize the grid environment and return the starting state.

        Args:
            chronics_id: Scenario index for deterministic initialization.
        """
        self._env.set_id(chronics_id % len(self._env.chronics_handler.subpaths))
        self._obs = self._env.reset()
        self._done = False
        self.reward = 0.0
        self._last_obs_data = self._obs_parser.parse(self._obs)
        return _format_obs_data(self._last_obs_data)

    def _step(self, action) -> str:
        """Step the environment and return the new observation string."""
        self._obs, _, self._done, _ = self._env.step(action)
        self.reward = composite_reward(self._obs, self._done)
        if self._done:
            raise RuntimeError("Blackout: the grid has collapsed. Episode over.")
        self._last_obs_data = self._obs_parser.parse(self._obs)
        return _format_obs_data(self._last_obs_data)

    def do_nothing(self) -> str:
        """Take no action and observe the grid's next state.

        Returns:
            The grid state after one timestep with no intervention.
        """
        return self._step(self._env.action_space({}))

    def set_line_status(self, line_id: int, status: str) -> str:
        """Connect or disconnect a powerline.

        Args:
            line_id: Index of the powerline to control (0 to n_lines-1).
            status: Either "connect" or "disconnect".

        Returns:
            The grid state after applying the action.
        """
        if not 0 <= line_id < self._env.n_line:
            raise ValueError(f"line_id {line_id} out of range [0, {self._env.n_line})")
        status_val = 1 if status == "connect" else -1
        action = self._env.action_space({"set_line_status": [(line_id, status_val)]})
        return self._step(action)

    def change_bus(self, element_type: str, element_id: int, bus: int) -> str:
        """Change the busbar assignment of a grid element at its substation.

        Args:
            element_type: One of "load", "gen", "line_or", or "line_ex".
            element_id: Index of the element within its type.
            bus: Target busbar (1 or 2).

        Returns:
            The grid state after applying the action.
        """
        key_map = {
            "load": "loads_id",
            "gen": "generators_id",
            "line_or": "lines_or_id",
            "line_ex": "lines_ex_id",
        }
        element_key = key_map.get(element_type)
        if element_key is None:
            raise ValueError(f"Unknown element_type: {element_type}")
        action = self._env.action_space(
            {"set_bus": {element_key: [(element_id, bus)]}}
        )
        return self._step(action)

    def redispatch(self, gen_id: int, delta_mw: float) -> str:
        """Adjust the power output target of a generator.

        Args:
            gen_id: Index of the controllable generator (0 to n_gens-1).
            delta_mw: MW adjustment — positive increases, negative decreases output.

        Returns:
            The grid state after applying the action.
        """
        if not 0 <= gen_id < self._env.action_space.n_gen:
            raise ValueError(f"gen_id {gen_id} out of range [0, {self._env.action_space.n_gen})")
        redispatch = [0.0] * self._env.action_space.n_gen
        redispatch[gen_id] = delta_mw
        action = self._env.action_space({"redispatch": redispatch})
        return self._step(action)

    def curtail(self, gen_id: int, max_mw: float) -> str:
        """Set an upper bound on a renewable generator's output.

        Args:
            gen_id: Index of the renewable generator (0 to n_gens-1).
            max_mw: Maximum allowed generation in MW.

        Returns:
            The grid state after applying the action.
        """
        if not 0 <= gen_id < self._env.action_space.n_gen:
            raise ValueError(f"gen_id {gen_id} out of range [0, {self._env.action_space.n_gen})")
        curtail = [-1.0] * self._env.action_space.n_gen
        curtail[gen_id] = max_mw
        action = self._env.action_space({"curtail": curtail})
        return self._step(action)

    def storage(self, storage_id: int, mw: float) -> str:
        """Control a battery storage unit.

        Args:
            storage_id: Index of the storage unit (0 to n_storage-1).
            mw: Power setpoint in MW — positive charges, negative discharges.

        Returns:
            The grid state after applying the action.
        """
        if not 0 <= storage_id < self._env.action_space.n_storage:
            raise ValueError(f"storage_id {storage_id} out of range [0, {self._env.action_space.n_storage})")
        storage_arr = [0.0] * self._env.action_space.n_storage
        storage_arr[storage_id] = mw
        action = self._env.action_space({"set_storage": storage_arr})
        return self._step(action)
