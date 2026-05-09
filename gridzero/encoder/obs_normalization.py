"""Per-feature observation normalization using precomputed statistics."""
from __future__ import annotations

import logging

import grid2op
import numpy as np

logger = logging.getLogger(__name__)


def compute_obs_stats(
    env_name: str,
    n_chronics: int = 100,
    steps_per_chronic: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample observations from the environment and compute per-feature mean/std.

    Takes random actions to explore diverse grid states (bus changes,
    redispatch, line disconnections) rather than just do-nothing trajectories.

    Args:
        env_name: grid2op environment name.
        n_chronics: number of chronics to sample from.
        steps_per_chronic: timesteps per chronic.
        seed: random seed for reproducibility.

    Returns:
        (mean, std) arrays of shape [flat_dim], computed across all samples.
    """
    rng = np.random.RandomState(seed)
    env = grid2op.make(env_name)
    act_space = env.action_space
    total_chronics = len(env.chronics_handler.real_data.subpaths)

    chronic_ids = rng.choice(total_chronics, size=min(n_chronics, total_chronics), replace=False)

    vecs: list[np.ndarray] = []
    for cid in chronic_ids:
        env.set_id(int(cid))
        obs = env.reset()
        vecs.append(obs.to_vect())

        for _ in range(steps_per_chronic):
            action = _random_action(act_space, obs, rng)
            obs, _, done, _ = env.step(action)
            if done:
                break
            vecs.append(obs.to_vect())

    env.close()

    data = np.stack(vecs, axis=0)
    mean = data.mean(axis=0).astype(np.float32)
    std = data.std(axis=0).astype(np.float32)

    n_const = (std < 1e-6).sum()
    logger.info(
        "Obs normalization: %d samples, %d features, %d constant (will be zeroed)",
        len(vecs), mean.shape[0], n_const,
    )

    return mean, std


def _random_action(act_space, obs, rng):
    """Generate a random valid action to explore diverse grid states."""
    choice = rng.randint(5)

    if choice == 0:
        return act_space({})

    if choice == 1 and act_space.n_line > 0:
        line_id = rng.randint(act_space.n_line)
        status = 1 if not obs.line_status[line_id] else -1
        return act_space({"set_line_status": [(line_id, status)]})

    if choice == 2 and act_space.n_sub > 0:
        sub_id = rng.randint(act_space.n_sub)
        bus = rng.randint(1, 3)
        elements = act_space.get_all_unitary_topologies_set(act_space, sub_id)
        if elements:
            return rng.choice(elements)
        return act_space({})

    if choice == 3 and act_space.n_gen > 0:
        gen_id = rng.randint(act_space.n_gen)
        ramp = obs.gen_max_ramp_up[gen_id]
        if ramp > 0:
            delta = rng.uniform(-ramp, ramp)
            redispatch = [0.0] * act_space.n_gen
            redispatch[gen_id] = delta
            return act_space({"redispatch": redispatch})

    return act_space({})


def log_feature_ranges(env_name: str, mean: np.ndarray, std: np.ndarray) -> None:
    """Log per-attribute normalization stats alongside physical limits."""
    env = grid2op.make(env_name)
    obs = env.reset()

    offset = 0
    lines: list[str] = []
    lines.append(
        f"{'Attribute':30s} {'Dim':>4s} {'Phys Lo':>10s} {'Phys Hi':>10s}"
        f" {'Samp Mean':>10s} {'Samp Std':>10s} {'Status':>10s}"
    )
    lines.append("-" * 100)

    physical_bounds = _get_physical_bounds(obs)

    for attr_nm in obs.attr_list_vect:
        val = np.array(getattr(obs, attr_nm)).ravel()
        dim = val.shape[0]
        chunk_mean = mean[offset:offset + dim]
        chunk_std = std[offset:offset + dim]

        phys = physical_bounds.get(attr_nm)
        if phys is not None:
            lo_str = f"{np.min(phys[0]):.2f}"
            hi_str = f"{np.max(phys[1]):.2f}"
        else:
            lo_str = "—"
            hi_str = "—"

        if dim == 0:
            lines.append(f"{attr_nm:30s} {dim:4d} {'—':>10s} {'—':>10s} {'(empty)':>10s} {'':>10s} {'—':>10s}")
            offset += dim
            continue

        avg_std = chunk_std.mean()
        status = "CONSTANT" if avg_std < 1e-6 else "ok"

        lines.append(
            f"{attr_nm:30s} {dim:4d} {lo_str:>10s} {hi_str:>10s}"
            f" {chunk_mean.mean():10.3f} {avg_std:10.3f} {status:>10s}"
        )
        offset += dim

    env.close()
    logger.info("Feature normalization summary:\n%s", "\n".join(lines))


def _get_physical_bounds(obs) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract known physical bounds from a grid2op observation."""
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    n_gen = len(obs.gen_p)
    n_line = len(obs.rho)

    bounds["year"] = (np.array([0]), np.array([9999]))
    bounds["month"] = (np.array([1]), np.array([12]))
    bounds["day"] = (np.array([1]), np.array([31]))
    bounds["hour_of_day"] = (np.array([0]), np.array([23]))
    bounds["minute_of_hour"] = (np.array([0]), np.array([59]))
    bounds["day_of_week"] = (np.array([0]), np.array([6]))

    bounds["gen_p"] = (obs.gen_pmin, obs.gen_pmax)
    bounds["rho"] = (np.zeros(n_line), np.full(n_line, np.inf))
    bounds["line_status"] = (np.zeros(n_line), np.ones(n_line))
    bounds["topo_vect"] = (np.full(len(obs.topo_vect), -1), np.full(len(obs.topo_vect), 2))
    bounds["curtailment_limit"] = (np.zeros(n_gen), np.ones(n_gen))

    if hasattr(obs, "thermal_limit"):
        bounds["a_or"] = (np.zeros(n_line), obs.thermal_limit)
        bounds["a_ex"] = (np.zeros(n_line), obs.thermal_limit)

    return bounds
