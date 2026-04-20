"""Reward functions for grid2op environments."""
from __future__ import annotations

import numpy as np


def survival_reward(done: bool, max_steps: int = 2016) -> float:
    """Reward for surviving one timestep; large penalty on blackout.

    Returns +1/max_steps per step survived, -1.0 on terminal blackout.
    Summed over an episode this gives +1.0 for a perfect run and approaches
    -1.0 the earlier the agent causes a blackout.
    """
    return -1.0 if done else 1.0 / max_steps


def load_served_ratio(obs) -> float:
    """Fraction of total load currently served (0.0 to 1.0).

    Uses obs.load_p (active power demanded) as the reference. If a load is
    disconnected, it contributes 0 to the numerator.
    """
    total = float(np.sum(obs.load_p))
    if total == 0.0:
        return 1.0
    # TODO: grid2op provides actual_dispatch and load disconnection info;
    # for now approximate with rho-based heuristic
    return float(np.clip(1.0 - np.mean(np.maximum(obs.rho - 1.0, 0.0)), 0.0, 1.0))


def line_capacity_margin(obs) -> float:
    """Mean spare thermal capacity across all lines.

    Returns 1 - mean(rho), clamped to [0, 1]. Higher is better; 0 means at
    least one line is at or over its thermal limit on average.
    """
    return float(np.clip(1.0 - float(np.mean(obs.rho)), 0.0, 1.0))


def composite_reward(
    obs,
    done: bool,
    max_steps: int = 2016,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted combination of reward components.

    Default weights: survival=1.0, load_served=0.5, line_margin=0.2.
    """
    if weights is None:
        weights = {"survival": 1.0, "load_served": 0.5, "line_margin": 0.2}

    r = (
        weights.get("survival", 1.0) * survival_reward(done, max_steps)
        + weights.get("load_served", 0.5) * load_served_ratio(obs)
        + weights.get("line_margin", 0.2) * line_capacity_margin(obs)
    )
    return float(r)
