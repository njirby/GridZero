# bench/ — grid2op LLM benchmarking instrument

Measures how well **AA-Dense-Blackwell** operates the grid2op power grid.
Headline metric: **improvement over do-nothing** (DoNothing pinned to 0) +
**safety** (survival, trips), with **cost** (tokens / wall) as a secondary axis.
Full design + rationale in **`../PLAN-BENCH.md`**.

> **Thinking time is unlimited per step** (skill, not speed). No per-turn or
> per-episode deadline; the model may read docs / inspect state as long as it
> needs. A generous safety cap (default 12 h/ep) only checkpoints a wedged run.

## Quick start

```bash
cd /home/nate/Documents/GridZero

make bench-validate     # Phase 0 gate: prove the metric on free baselines (no LLM spend)
make bench-baselines    # run the free baseline ladder over the standard panel
make bench-pilot        # LLM over the 6 pilot chronics @96, concurrency 2  (hours)
make bench-report       # aggregate everything into a leaderboard (MD + HTML + JSON)
```

## Files

| file | what |
|---|---|
| `score.py` | `EpisodeResult` + `normalize()` (DN=0 anchor, L2RPN [−100,0,80,100] scale) + `self_check()`. **Single source of truth for scoring.** |
| `panel.json` | the fixed, versioned evaluation panel (chronics, horizons, sampling, safety). Changing it = a new benchmark; bump `version`. |
| `panel.py` | load panel + `config_hash()` (reproducibility: model+env+panel+horizon+sample). |
| `agents.py` | `N1GreedyAgent` — the preventive expert baseline (N-1 simulate, acts only when a trip is imminent). |
| `run_baselines.py` | run the free ladder (DoNothing/Random/RecoPowerline/N1Greedy/…) over any panel/horizon → `runs/bench-<ts>/baselines.jsonl`. |
| `validate_metric.py` | Phase 0 gate: metric invariants + a discriminating-horizon check (DN=0, Random=floor, expert≈DN). |
| `run_llm.py` | run ONE long-horizon LLM episode (own backend + opencode session, own ports). Robust: unlimited thinking, gentle auto-continue on idle, append-only trace, safety cap. |
| `run_pilot.py` | the LLM over all pilot chronics in parallel (bounded concurrency). |
| `report.py` | aggregate baselines + LLM results → leaderboard (bootstrap CIs, per-chronic detail, efficiency axis, rule-based error taxonomy) as MD + HTML + JSON. |

## The baseline ladder

| agent | speed | role |
|---|---|---|
| `DoNothing` | ~6 ms/step | the **0 anchor** |
| `Random` | ~6 ms/step | the **floor** |
| `RecoPowerline` | fast | recovery expert (reconnect trips) — ties DN on the sandbox (no opponent) |
| `N1Greedy` (ours) | ~45 ms/step | preventive expert — the honest "smart non-LLM" bar |
| `TopologyGreedy` / `AlertAgent` | **~14 s/step** | too slow for per-episode runs (cited only) |

## Running a single LLM episode

```bash
./.venv/bin/python bench/run_llm.py --chronic 0 --horizon 96 --port 8820 \
    --repeats 1 --safety-cap-h 10 --poke-idle-s 240
```
- Each episode = own backend (port) + own opencode session (port+200) → isolated,
  so episodes parallelize cleanly (see `run_pilot.py`).
- Each episode runs in its own network namespace (`bench/netns.py`); the sandbox can
  reach only the vLLM forwarder port on the gateway IP. If netns setup fails the
  episode FAILS; `--allow-no-netns` overrides it and the result records
  `isolated=false`. While an episode runs, `runs/backend-bench-<port>.json` holds
  its netns address + operator token for `scripts/bench_watch.py`.
- Output: `runs/llm-<chronic>-<ts>/results.json` (EpisodeResult list) + `config.json`
  (with `config_hash`) + the append-only trace at `runs/<ep>.jsonl`.

## Horizons (why three)

- **pilot 96** — cheap first signal. On the sandbox DN doesn't trip by 96, so
  survival is null and the read is "did it operate safely + find any redispatch
  gain." Fast-ish.
- **standard 1200** — the real benchmark. DN trips on every pilot chronic by
  ~800–1097, so a competent agent must *outlast passive* and beat it on reward.
- **baselines_full 8064** — full chronic length, run FREE (baselines) to pin DN
  anchors across the whole chronic; reserved for LLM robustness runs.

## Reproducibility

- `config_hash` (model id + env + panel version + horizon + sampling + harness
  version + ablation config + seed + adversarial params) is recorded on every run → results comparable + auditable across time.
- `reset(options={"time serie id": k, "max step": H})` + `reset(seed=s)` pin the
  chronic, horizon, and RNG (verified on grid2op 1.12.5).
- Panel chronic ids are **hidden from the model** (docs/AGENTS.md never mention
  them) → no overfit to the eval set.
- opencode runs at **temp 0** (primary) → reproducible pass@1; k=3 repeats →
  bootstrap CIs; a temp>0 probe → variance.
- Full event **traces are append-only** (`runs/<ep>.jsonl`) → any episode is
  replayable/auditable, and a crash checkpoints rather than corrupts.

## Cost model

The gateway reports `$0` (cost tracking off), so **tokens are the real cost
metric**. Observed (smoke, 8 steps): ~2.4M cumulative input tokens (each turn
resends growing context; opencode auto-compacts as it fills the 1M window),
~13k output, ~24 min (mostly one-time doc/source exploration). Steady-state
thinking ~30–180 s/step.
