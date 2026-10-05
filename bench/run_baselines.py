#!/usr/bin/env python
"""bench/run_baselines.py — run the free baseline ladder over the panel.

Agents (all ship with grid2op; same act(obs, reward, done) interface the LLM's
`simctl act` maps to, so they are directly comparable):
  DoNothing        -> the 0 anchor (passive reference)
  Random           -> the floor
  RecoPowerline    -> rule-based expert (reconnect tripped lines, simulated)
  TopologyGreedy   -> rule-based expert (reconfigure substations)
  AlertAgent       -> rule-based expert (reconnect + alert)

Usage:
  ./.venv/bin/python bench/run_baselines.py [--pilot|--standard|--full] [--agents a,b,c] [--seed 0]
Writes runs/bench-<ts>/baselines.jsonl (one EpisodeResult per line) + prints a
normalized table. DoNothing on the same (chronic, horizon) is the anchor for
each row, so rows are comparable even on chronics where DN itself trips.
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import grid2op
from grid2op.Agent import (DoNothingAgent, RandomAgent, RecoPowerlineAgent,
                           TopologyGreedy, AlertAgent)
from bench.score import EpisodeResult, normalize
from bench.agents import N1GreedyAgent

AGENTS = {
    "DoNothing": DoNothingAgent,
    "Random": RandomAgent,
    "RecoPowerline": RecoPowerlineAgent,   # fast; recovery-only (ties DN on sandbox)
    "N1Greedy": N1GreedyAgent,             # preventive expert (see bench/agents.py)
    "TopologyGreedy": TopologyGreedy,      # ~14 s/step -> only for short horizons
    "AlertAgent": AlertAgent,
}
PANEL = json.load(open(os.path.join(ROOT, "bench", "panel.json")))


def run_episode(env, ag_cls, name, chronic, horizon, seed):
    """Run one full episode (to horizon or game-over). env.step returns done=True
    in BOTH cases, so 'survived' is defined by reaching the horizon: t >= horizon."""
    t0 = time.time()
    obs = env.reset(options={"time serie id": chronic, "max step": horizon})
    ag = ag_cls(env.action_space)
    ag.seed(seed)
    r, done = env.current_reward, False
    cum, t = 0.0, 0
    n_trips = n_illegal = n_ambiguous = 0
    peak_rho = 0.0
    while not done:
        a = ag.act(obs, r, done)
        obs, r, done, info = env.step(a)
        t = int(obs.current_step)
        cum += float(r)
        disc = np.asarray(info.get("disc_lines", []), dtype=int)
        n_trips += int(sum(1 for j in disc if j >= 0))
        n_illegal += int(bool(info.get("is_illegal")))
        n_ambiguous += int(bool(info.get("is_ambiguous")))
        rho = np.asarray(obs.rho, dtype=float)
        if rho.size:
            peak_rho = max(peak_rho, float(np.nanmax(rho)))
    survived = t >= horizon
    return EpisodeResult(
        agent=name, chronic=chronic, horizon=horizon, survived=t,
        done=survived, cum_reward=float(cum), n_trips=n_trips,
        n_illegal=n_illegal, n_ambiguous=n_ambiguous, peak_rho=round(peak_rho, 4),
        n_down_final=int(np.sum(~np.asarray(obs.line_status, dtype=bool))),
        game_over=not survived, wall_clock_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--pilot", action="store_true")
    grp.add_argument("--standard", action="store_true")
    grp.add_argument("--full", action="store_true", help="baselines at full 8064 horizon")
    ap.add_argument("--agents", default="DoNothing,Random,RecoPowerline,N1Greedy")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.full:
        chronics, horizon = PANEL["pilot_chronics"], PANEL["horizons"]["baselines_full"]
    elif a.standard:
        chronics, horizon = PANEL["standard_chronics"], PANEL["horizons"]["standard"]
    else:
        chronics, horizon = PANEL["pilot_chronics"], PANEL["horizons"]["pilot"]
    names = [n.strip() for n in a.agents.split(",") if n.strip()]
    assert "DoNothing" in names, "DoNothing (the anchor) must be in --agents"

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(ROOT, "runs", f"bench-{ts}")
    os.makedirs(outdir, exist_ok=True)
    env = grid2op.make(PANEL["env"])

    results = {}   # (agent, chronic) -> EpisodeResult
    t0 = time.time()
    for name in names:
        cls = AGENTS[name]
        for k in chronics:
            res = run_episode(env, cls, name, k, horizon, seed=a.seed)
            results[(name, k)] = res
            print(f"  {name:16s} chronic={k:4d} survived={res.survived:5d}/{horizon} "
                  f"cum={res.cum_reward:9.0f} trips={res.n_trips} done={res.done}", flush=True)
    print(f"(baselines wall {time.time()-t0:.0f}s)")

    # write + normalize against per-chronic DN anchors
    rows = []
    with open(os.path.join(outdir, "baselines.jsonl"), "w") as f:
        for (name, k), res in results.items():
            dn = results[("DoNothing", k)]
            sc = normalize(res, dn)
            row = {**res.to_dict(), **sc}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
    # table
    print(f"\n=== {PANEL['env']} horizon={horizon} seed={a.seed} — normalized vs DN ===")
    print(f"{'agent':18s} {'norm(mean)':>10s} {'improvement(mean)':>18s} {'survival%':>10s} {'trips(mean)':>11s}")
    for name in names:
        rs = [r for r in rows if r["agent"] == name]
        print(f"{name:18s} {np.mean([r['norm'] for r in rs]):10.2f} "
              f"{np.mean([r['improvement'] for r in rs]):18.4f} "
              f"{100*np.mean([r['done'] for r in rs]):10.1f} "
              f"{np.mean([r['n_trips'] for r in rs]):11.2f}")
    print(f"\nwrote {outdir}/baselines.jsonl ({len(rows)} rows)")


if __name__ == "__main__":
    main()
