"""Smoke tests for grid2op wrapper and observation parsing."""
import pytest


def test_import_actions_without_grid2op():
    """Schema-only imports should work without hard grid2op dependency."""
    from gridzero.env.actions import get_json_schema

    schema = get_json_schema()
    assert isinstance(schema, dict)


def test_make_env_smoke():
    """make_env returns a grid2op env and ObsParser without error."""
    pytest.importorskip("grid2op")
    from omegaconf import OmegaConf
    from gridzero.env.wrapper import make_env

    cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "l2rpn_case14_sandbox",
                "backend": "lightsim2grid",
                "max_steps": 100,
                "test": True,
            }
        }
    )
    env, obs_parser = make_env(cfg)
    assert env is not None
    assert obs_parser is not None
    env.close()


def test_obs_parser_parse():
    """ObsParser.parse() returns ObsData with flat vector and raw observation."""
    pytest.importorskip("grid2op")
    from omegaconf import OmegaConf
    from gridzero.env.wrapper import make_env
    from gridzero.env.observation import ObsData

    cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "l2rpn_case14_sandbox",
                "backend": "lightsim2grid",
                "max_steps": 10,
                "test": True,
            }
        }
    )
    env, obs_parser = make_env(cfg)
    obs = env.reset()
    parsed = obs_parser.parse(obs)
    assert isinstance(parsed, ObsData)
    assert parsed.flat.ndim == 1
    assert parsed.raw is not None
    env.close()
