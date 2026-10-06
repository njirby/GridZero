"""Environment construction helper for grid2op environments."""
from __future__ import annotations

import grid2op
from omegaconf import DictConfig

from gridzero.env.observation import ObsParser


def make_env(cfg: DictConfig):
    """Construct a grid2op env and an ObsParser for it.

    Returns:
        (env, obs_parser) where env is a grid2op Environment.
    """
    backend = _get_backend(cfg)
    env = grid2op.make(
        cfg.env.env_name,
        backend=backend(),
        test=bool(cfg.env.get("test", False)),
    )
    obs_parser = ObsParser(env)
    return env, obs_parser


def _get_backend(cfg: DictConfig):
    backend_name = cfg.env.get("backend", "lightsim2grid")
    if backend_name == "lightsim2grid":
        from lightsim2grid import LightSimBackend
        return LightSimBackend
    from grid2op.Backend import PandaPowerBackend
    return PandaPowerBackend
