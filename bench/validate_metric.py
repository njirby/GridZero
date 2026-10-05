#!/usr/bin/env python
"""bench/validate_metric.py — Phase 0 gate: prove the metric behaves BEFORE any
LLM spend.

1. Offline property checks of bench.score (no env).
2. Fast smoke: DoNothing + RecoPowerline on 2 chronics at 288 steps (runner plumbing).
3. Full validation: DoNothing, Random, RecoPowerline, N1Greedy on 2 pilot
   chronics at 1200 steps — a horizon where do-nothing TRIPS on every pilot
   chronic (calibrated survival: 381..1097 < 1200). Assertions (the true
   invariants of the metric):
     - DN improvement == 0 and DN norm == 0 on every chronic (the anchor)
     - Random mean norm < DN mean norm (the floor)
     - expert (N1Greedy) mean norm >= DN - epsilon (not catastrophically worse)
     - the game-over survival branch produces (-100, 0) for an early death
     FINDING: on this sandbox the preventive expert ties DN (die-time is set by
     the natural cascade, not by late trip-mitigation). The discriminating axis
     for a capable agent is REDISPATCH loss-optimization (RedispReward pays for
     it) — that is what the LLM must beat zero on.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from bench import score
from bench.run_baselines import run_episode, AGENTS, PANEL
from bench.agents import N1GreedyAgent
AGENTS["N1Greedy"] = N1GreedyAgent


def main():
    print("== 1. offline property checks ==")
    score.self_check()

    import grid2op
    env = grid2op.make(PANEL["env"])
    chronics = PANEL["pilot_chronics"][:2]

    print("== 2. smoke @288 (plumbing) ==")
    smoke = {}
    for name in ["DoNothing", "RecoPowerline"]:
        for k in chronics:
            r = run_episode(env, AGENTS[name], name, k, 288, seed=0)
            smoke[(name, k)] = r
            print(f"  {name:14s} k={k}: survived {r.survived}, cum {r.cum_reward:.0f}, done {r.done}")

    print("== 3. validation @1200 (DN trips on all pilot chronics) ==")
    H = 1200
    res = {}
    for name in ["DoNothing", "Random", "RecoPowerline", "N1Greedy"]:
        for k in chronics:
            r = run_episode(env, AGENTS[name], name, k, H, seed=0)
            res[(name, k)] = r
            dn = res[("DoNothing", k)]
            sc = score.normalize(r, dn)
            print(f"  {name:14s} k={k}: surv {r.survived:4d}/{H} cum {r.cum_reward:8.0f} trips {r.n_trips} "
                  f"norm {sc['norm']:7.2f} impr {sc['improvement']:+.4f}")

    # --- invariants ---
    dn_imp = [score.normalize(res[("DoNothing", k)], res[("DoNothing", k)])["improvement"] for k in chronics]
    dn_norm = [score.normalize(res[("DoNothing", k)], res[("DoNothing", k)])["norm"] for k in chronics]
    assert all(abs(x) < 1e-9 for x in dn_imp), "DN anchor must be 0 improvement"
    assert all(abs(x) < 1e-9 for x in dn_norm), "DN (not completing) must be 0 norm"
    assert all(res[("DoNothing", k)].game_over for k in chronics), "DN must game-over @1200 (calibrated)"

    ra_norm = np.mean([score.normalize(res[("Random", k)], res[("DoNothing", k)])["norm"] for k in chronics])
    assert ra_norm < -50.0, f"random must be the floor, got {ra_norm}"

    ex_norm = np.mean([score.normalize(res[("N1Greedy", k)], res[("DoNothing", k)])["norm"] for k in chronics])
    assert ex_norm >= -5.0, f"preventive expert must not be catastrophically worse than DN, got {ex_norm}"

    early = score.EpisodeResult(agent="early", chronic=0, horizon=H, survived=100, done=False,
                                cum_reward=800.0, game_over=True)
    assert -100.0 < score.normalize(early, res[("DoNothing", 0)])["norm"] < 0.0
    print(f"\n  DN anchor: 0.0 | Random floor: {ra_norm:.2f} | N1Greedy expert: {ex_norm:.2f}")
    print("\nVALIDATE METRIC: PASS")
    print("FINDING: on this sandbox the preventive expert ties DN — the die-time is set")
    print("by the natural cascade, not late trip-mitigation. The discriminating axis for")
    print("a capable agent is REDISPATCH loss-optimization (RedispReward pays for it).")


if __name__ == "__main__":
    main()
