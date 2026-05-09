"""Tests for observation serialization and reward utilities."""
import json

import numpy as np
import pytest


def test_obs_to_prompt_is_string():
    from gridzero.env.observation import ObsData
    from gridzero.env.serialization import obs_to_prompt

    obs = ObsData(
        flat=np.zeros(100, dtype=np.float32),
        graph=None,
        n_lines=5, n_loads=3, n_gens=2, n_substations=4,
    )
    obs.rho = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    obs.line_status = np.array([True] * 5)
    obs.load_p = np.array([10.0, 12.0, 8.0])
    obs.gen_p = np.array([20.0, 15.0])
    obs.v_or = np.array([1.0] * 5)

    prompt = obs_to_prompt(obs)
    assert isinstance(prompt, str)
    assert "assistant" in prompt
    assert "rho" in prompt


def test_obs_to_dataset_row_has_prompt():
    from gridzero.env.observation import ObsData
    from gridzero.env.serialization import obs_to_dataset_row

    obs = ObsData(
        flat=np.zeros(50, dtype=np.float32),
        graph=None,
        n_lines=3, n_loads=2, n_gens=1, n_substations=3,
    )
    row = obs_to_dataset_row(obs, env_id=42)
    assert "prompt" in row
    assert row["env_id"] == 42
    assert isinstance(row["obs_flat"], list)


def test_composite_reward_components():
    from gridzero.rewards.grid_rewards import composite_reward

    class FakeObs:
        rho = np.array([0.5, 0.6])
        load_p = np.array([10.0, 12.0])

    r = composite_reward(FakeObs(), done=False)
    assert isinstance(r, float)
    assert r > 0

    r_done = composite_reward(FakeObs(), done=True)
    assert r_done < r
