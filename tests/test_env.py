"""Smoke tests for grid2op wrapper and observation parsing."""
import pytest


def test_make_env_smoke():
    """make_env returns a GymEnv and ObsParser without error."""
    pytest.importorskip("grid2op")
    from omegaconf import OmegaConf
    from gridzero.env.wrapper import make_env

    cfg = OmegaConf.create({
        "env": {
            "env_name": "l2rpn_case14_sandbox",
            "backend": "lightsim2grid",
            "max_steps": 100,
        }
    })
    gym_env, obs_parser = make_env(cfg)
    assert gym_env is not None
    assert obs_parser is not None
    gym_env.close()


def test_obs_parser_parse():
    """ObsParser.parse() returns ObsData with a flat vector."""
    pytest.importorskip("grid2op")
    import grid2op
    from omegaconf import OmegaConf
    from gridzero.env.wrapper import make_env
    from gridzero.env.observation import ObsData

    cfg = OmegaConf.create({
        "env": {
            "env_name": "l2rpn_case14_sandbox",
            "backend": "lightsim2grid",
            "max_steps": 10,
        }
    })
    gym_env, obs_parser = make_env(cfg)
    raw_obs, _ = gym_env.reset()
    assert isinstance(raw_obs, ObsData)
    assert raw_obs.flat is not None
    assert raw_obs.flat.ndim == 1
    gym_env.close()
