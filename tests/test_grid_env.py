"""Tests for the TRL-compatible GridEnv class."""
from __future__ import annotations

import inspect
import json

import pytest

from gridzero.training.env import GridEnv


@pytest.fixture
def env():
    e = GridEnv()
    yield e
    e._env.close()


def test_reset_returns_json_string(env):
    obs_str = env.reset(chronics_id=0)
    data = json.loads(obs_str)
    assert "rho" in data
    assert "line_status" in data
    assert "n_lines" in data
    assert "n_gens" in data
    assert isinstance(data["rho"], list)


def test_deterministic_reset():
    env1 = GridEnv()
    env2 = GridEnv()
    obs1 = env1.reset(chronics_id=7)
    obs2 = env2.reset(chronics_id=7)
    assert obs1 == obs2
    env1._env.close()
    env2._env.close()


def test_do_nothing_returns_obs_and_sets_reward(env):
    env.reset(chronics_id=0)
    obs_str = env.do_nothing()
    data = json.loads(obs_str)
    assert "rho" in data
    assert isinstance(env.reward, float)


def test_set_line_status_changes_observation(env):
    env.reset(chronics_id=0)
    obs_before = env.do_nothing()
    env.reset(chronics_id=0)
    obs_after = env.set_line_status(line_id=0, status="disconnect")
    assert obs_before != obs_after


def test_redispatch_sets_reward(env):
    env.reset(chronics_id=0)
    env.redispatch(gen_id=0, delta_mw=1.0)
    assert isinstance(env.reward, float)


def test_invalid_element_type_raises(env):
    env.reset(chronics_id=0)
    with pytest.raises(ValueError, match="Unknown element_type"):
        env.change_bus(element_type="invalid", element_id=0, bus=1)


def test_all_tool_methods_have_docstrings():
    """TRL requires docstrings with Args: blocks to generate tool schemas."""
    env = GridEnv()
    public_methods = [
        name for name in dir(env)
        if not name.startswith("_") and name != "reset" and callable(getattr(env, name))
    ]
    assert len(public_methods) == 6, f"Expected 6 tool methods, got {public_methods}"

    for name in public_methods:
        method = getattr(env, name)
        doc = inspect.getdoc(method)
        assert doc is not None, f"{name} has no docstring"
        assert "Args:" in doc or name == "do_nothing", f"{name} docstring missing Args: block"

    env._env.close()


def test_reward_resets_on_new_episode(env):
    env.reset(chronics_id=0)
    env.do_nothing()
    first_reward = env.reward
    env.reset(chronics_id=0)
    assert env.reward == 0.0
    assert first_reward != 0.0
