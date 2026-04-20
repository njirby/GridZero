"""Tests for GSPO-related utilities.

GSPO training itself is handled by ms-swift (GRPOTrainer +
importance_sampling_level='sequence'). These tests cover the ORM plugin
and the observation serialization that feeds ms-swift's dataset pipeline.
"""
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
    # Attach minimal attributes that serialization tries to read
    obs.rho = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    obs.line_status = np.array([True] * 5)
    obs.load_p = np.array([10.0, 12.0, 8.0])
    obs.gen_p = np.array([20.0, 15.0])
    obs.v_or = np.array([1.0] * 5)

    prompt = obs_to_prompt(obs)
    assert isinstance(prompt, str)
    assert "assistant" in prompt  # chat template present
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


def test_orm_plugin_registers():
    """Importing the ORM plugin should register 'grid_composite' in orms."""
    pytest.importorskip("swift")
    from swift.plugin import orms  # type: ignore[import]
    import gridzero.rewards.orm_plugin  # noqa: F401
    assert "grid_composite" in orms
