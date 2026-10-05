# Benchmark Plan — scoring how well AA-Dense-Blackwell runs the grid

**Goal:** a rigorous, reproducible instrument that measures how well the model
operates a grid2op power grid — headline = *improvement over do-nothing* (DN
pinned to 0) + *safety* (survival/trips), with *cost* (tokens/$/wall) as a
secondary axis. Built on the working `grid2op-harness` (simctl + opencode +
backend). Scope (approved): single-model deep profile, score+safety lead,
pilot-then-scale, cooperative-stability now / adversarial later.

**Thinking-time policy (approved):** UNLIMITED per step. No per-turn or
per-episode deadline. The model may read docs / inspect state / look around for
as long as it needs on each step. Wall-clock + turns are *reported* (cost axis),
never a scored deadline — so the benchmark measures *skill*, cleanly separated
from speed. A generous safety cap (default 12 h/episode) only checkpoints a
wedged run; it does not corrupt anything (the trace is append-only on disk).

---

## 1. The metric (VALIDATED — Phase 0 passed)

The sandbox uses **`RedispReward`** (range −10…294.5, healthy passive step ≈ +64):
an operational-cost/"regret" reward where **higher is better** and it *explicitly
rewards `redispatch` to cut line losses* (α_redisp = 5.0). So the sandbox has a
real optimization signal, not just "don't crash."

**Canonical construction** (L2RPN 2020–2022 convention, from R1):
- `improvement` (HEADLINE) = `(S_A − S_DN) / |S_DN|` on the env's own `cum_reward`,
  where S = cumulative reward over a fixed (chronic, horizon). **DoNothing = 0.**
- `norm` (secondary, L2RPN [−100,0,80,100] scale): game-over →
  `−100·(1 − survival_frac)`; completed → `80 + 20·clamp(improvement/0.2, 0, 1)`.
- Both computed from **sim ground truth** (`cum_reward` + survival) — never from
  anything the model is told. `bench/score.py` is the single source of truth and
  passes `self_check()` + the full validation.

**Reference bars (R1, from L2RPN leaderboards):** do-nothing **0** ·
rule-based expert **~22** · strong RL **~45…61** (PowRL 61.48 = open-board top).

### Key empirical finding (calibrated on the sandbox)
Do-nothing **trips out in only 5–14% of the full 8064-step chronics**
(survives 381–1097 steps; trips 5 lines). So:
- **@96 and @288 steps the sandbox is EASY — nothing trips, and DN is
  near-optimal** (redispatch headroom is tiny in the first hours). On these
  horizons a capable agent mostly *matches* DN; the signal is "did it operate
  safely without crashing" + marginal redispatch gain. (Confirmed: @8 the model
  no-oped, cum=513.937 = DN exactly, 0 trips, improvement≈0 — the *correct*
  local optimum, not a bug.)
- **@1200 steps (the standard horizon) is where it discriminates**: DN dies, so
  a competent agent must *outlast passive* (survival) **and** beat it on reward.
- **@8064 (full competition length)** = the robustness target; opencode
  auto-compacts context as it fills (1M window), so it's supported.

This shapes the run design (below): pilot=96 (cheap safety signal),
standard=1200 (real benchmark), baselines_full=8064 (pin DN anchors cheaply).

---

## 2. Baseline ladder (VALIDATED — all free in grid2op, same act() interface)

| agent | class | speed | role | @96 result |
|---|---|---|---|---|
| **DoNothing** | `DoNothingAgent` | ~6ms/step | the **0 anchor** | 0.00 |
| **Random** | `RandomAgent` | ~6ms/step | the **floor** | −0.97 |
| **RecoPowerline** | `RecoPowerlineAgent` | fast | recovery expert (reconnect trips) | 0.00 (ties DN — no opponent to recover from) |
| **N1Greedy** | `bench/agents.py` (ours) | ~45ms/step | **preventive expert** (N-1 simulate, acts only when a trip is imminent, commits only if strictly better) | 0.00 (honest: die-time is set by the cascade, not late mitigation) |
| TopologyGreedy / AlertAgent | shipped | **~14 s/step** | too slow for per-episode runs (~30 h/episode) | cited only |

**FINDING:** the shipped "expert" family is the wrong shape for this benchmark
(it's built for opponent-attack recovery, not loss optimization, and the greedy
ones are too slow). `N1Greedy` (ours) is the honest preventive expert — and it
*ties* DN, confirming that on the sandbox the discriminating axis is **redispatch
loss-optimization**, exactly what `RedispReward` pays for. That is the axis the
LLM must beat zero on.

All baselines run full-length (8064) in seconds–minutes (free, no LLM) → exact
per-chronic DN anchors for the whole chronic.

---

## 3. Run design

| | Pilot | Standard | Extended (opt) |
|---|---|---|---|
| Grid | case14_sandbox | case14_sandbox | wcci_2020 (36-sub), neurips-track1 (118-sub) |
| Horizon | **96** (8h) | **1200** (1 day-ish; DN trips here) | **8064** (28 days, full) |
| Chronic panel | **6** | **24** (mirrors official test-set size) | per-grid |
| Model repeats | k=1 | k=3 (temp 0 → CI) + a temp>0 probe | — |
| Thinking time | unlimited | unlimited | unlimited |
| Safety cap | 8 h/ep | 12 h/ep | 12 h/ep |

**Reproducibility (R2's #1 lesson — model drift has inverted studies):**
- Every run record carries a **config hash** (model id + env + panel version +
  horizon + sampling + harness version) — `bench/panel.py`.
- `reset(options={"time serie id": k, "max step": H})` pins the chronic + horizon;
  `reset(seed=s)` pins the RNG (verified). Panel chronic ids are **hidden from the
  model** (docs/AGENTS.md never mention them) → no overfit to the eval set.
- opencode runs at **temp 0** (primary) for reproducible pass@1; k=3 repeats for
  bootstrap CIs; a temp>0 probe for variance.
- **Full traces are append-only** (`runs/<ep>.jsonl`) → auditable + resumable.

---

## 4. Long-horizon robustness (the bar you set — VALIDATED via the @8 smoke)

The smoke run (horizon 8) drove the whole path and surfaced + fixed the issues
that would have killed a 8064-step run:

1. **Done-guard + C3 caching** (`sim_session.py`): after the episode is
   finished (t≥horizon or game-over), grid2op requires `reset()` before further
   env access. The session now caches the last good C3 and refuses to touch a
   terminal env — `step`/`act`/`stats`/`status` all return clean "episode over"
   instead of 500ing. (Backend suite still 25/25 after this.)
2. **`to_c3` never calls `env.get_thermal_limit()`** (that raises post-done;
   thermal limits live in `build_meta`).
3. **Auto-compaction** (opencode default, confirmed by your experience): context
   growth across hundreds/thousands of turns is opencode's job, not the harness'.
   The 1M window + compaction means full-competition runs are supported.
4. **Gentle auto-continue**: if the sim makes no progress for `--poke-idle-s`
   while not done, the runner sends a steer (queued at the model's next
   boundary — never interrupts a long think). This is the *only* pacing nudge.
5. **Append-only trace + per-episode backend + safety cap**: one crash never
   affects another (each episode = own backend on own port + own opencode
   session, persistent on disk); a 12-h cap checkpoints rather than corrupts.
6. **The model is honest**: it read the reward source, ran its own
   no-op-vs-redispatch experiment, and acted only where it measured a gain —
   good behavior, but it means we must *nudge* it toward proactive redispatch
   (see kickoff) and *not* count on it redispatching for free on easy horizons.

---

## 5. Code (what exists / what's built)

**Built + green:**
- `bench/score.py` — EpisodeResult + normalize() + self_check (VALIDATED).
- `bench/panel.json` v2 — fixed pilot/standard/full chronics + horizons.
- `bench/panel.py` — load panel + config_hash.
- `bench/agents.py` — N1GreedyAgent (preventive expert baseline).
- `bench/run_baselines.py` — runs the ladder over any panel/horizon →
  `runs/bench-<ts>/baselines.jsonl` (VALIDATED: DN=0, Random=−0.97, expert=0).
- `bench/validate_metric.py` — Phase 0 gate (PASSED).
- `bench/report.py` — leaderboard MD+HTML+JSON, bootstrap CIs, heatmap,
  efficiency axis, rule-based error taxonomy (works on baseline data).
- `backend/app/sim_session.py` — reset(seed, options={max step, time serie id})
  + episode_stats + done-guard + C3 caching (backend 25/25).
- `backend/app/opencode_driver.py` — session_stats() (cumulative tokens/cost).
- `backend/app/main.py` — `POST /bench/start`, `GET /bench/stats`.
- `bench/run_llm.py` — the robust long-horizon episode runner (VALIDATED @8:
  survived 8/8, 0 trips, wrote results.json, clean teardown).

**Makefile:** `bench-validate`, `bench-baselines`, `bench-pilot`, `bench-report`.

---

## 6. Execution phases

- **Phase 0 — metric spine: DONE.** Metric validated on free baselines; runner
  validated end-to-end (@8); report renders. No LLM spend to get here.
- **Phase 1 — pilot (RUNNING).** 6 chronics × 96 × {DN, Random, RecoPowerline,
  N1Greedy, **AA-Dense-Blackwell k=1**}. First real LLM read: does it operate
  safely (beat the game-over rate) and does it find any redispatch gain? ~8 h
  wall per episode (unlimited thinking), run in parallel across chronics.
- **Phase 2 — standard + ablations.** 24 chronics × 1200 × k=3 (the
  survival-discriminating horizon) + ablations: with/without docs, with/without
  the `render`/vision channel, compact vs `--detailed` observation. Headline
  leaderboard + CIs + ablation table + error taxonomy.
- **Phase 3 — difficulty + adversarial.** Extend to wcci_2020 / neurips-118
  (survival becomes the differentiator); add `grid2op.Opponent`
  (Geometric/WeightedRandom) → robustness mode; then full **2-agent
  defender-vs-attacker** scored by ELO/Bradley–Terry (defender: survival time,
  loss-of-load, time-to-cascade; attacker: damage). R2's methodology: no judge
  needed (ground truth in the sim).

---

## 7. Cost model (observed, not $ — gateway cost tracking is off)

The smoke: **8 steps ≈ 2.4M cumulative input tokens, 13k output, 24 min** (most
of it one-time doc/source exploration + heavy t=0 thinking). Steady-state is
~30–180 s/step of thinking. Standard 1200-step ≈ 20–40 h/episode at k=1. So:
- **tokens are the real cost metric** (we report tokens_in/out/reasoning).
- The one-time exploration (reading docs + reward source) is amortized per
  episode; a future optimization is a "primed" session that skips re-exploring.
- Full 8064-step runs are the expensive tail — reserve for the robustness
  target, not the default.

---

## 8. Anti-gaming (validity)

- Score from **sim ground truth**; model sees raw obs only.
- **Hidden chronics** — no chronic-specific tuning possible from docs.
- Model **excluded from the reward definition**.
- Full **traces archived + auditable** — a model that just no-ops to dodge
  illegal actions scores ≈0 improvement (caught: the @8 run did exactly this,
  honestly).
- Per-action constraints (max 1 line + 1 sub/step) enforced by the **sim**, not
  the harness.

---

## 9. Live findings (pilot, in progress)

**Port isolation (fixed):** each episode = own backend port + own opencode port
(simctl targets the right sim; parallel episodes don't cross-contaminate). A
stale backend on 8731 had been silently driving the wrong sim before this fix —
now each `run_llm.py` sets `SIMCTL_BACKEND_PORT` + `OPENCODE_PORT` per episode,
and the driver refuses to attach to a stale opencode server.

**The model operates with genuine grid reasoning** (chronic 0 @96, observed):
- First turn is a **~20 min / ~110k-token analysis**: reads the docs + reward
  source, reasons about generator margins and power flow ("sub_0's gen exports
  all of it; to reduce losses I want power from near gens to nearby loads").
- First act = an **informed net-zero redispatch** (`gen_0_5:−15, gen_2_1:+10,
  gen_1_0:+5`, summing to 0) to cut losses, not just avoid trips.
- Per-step pace **drops after the initial analysis** (~20 min first turn →
  ~3 min/step). The upfront exploration is a one-time cost per episode.
- Operates **safely**: 0 trips, cum tracking passive (~60/step).

**Findings worth carrying into the report:**
1. **Upfront exploration cost is real** (~110k tokens/episode reading docs +
   source before the first act). A "primed" session (pre-loaded context) would
   amortize it — a future optimization, not a metric confound.
2. **The observation interface is the baseline** for this benchmark. `observe`
   does not expose generator margins (the model reads them from docs). Keeping it
   constant across runs is required for comparability; "expose margins in observe"
   is a separate *ablation axis* for a later iteration, not a baseline change.
3. **No-vision / detailed-obs / no-docs ablations** are wired (`bench/configs/`,
   `--config`) and ready for Phase 2.

### First read (pilot @96) — **INVALIDATED by an answer-key leak** (2026-10-03)

The pilot ran with opencode's working dir = the full repo root, so the model could —
and did — read the benchmark's **answer key** by absolute path. Traces show it
`cat`-ed `bench/score.py` (the exact metric), `bench/panel.json` (the chronic ids →
"hidden" chronics were not hidden), `bench/agents.py` (the N1Greedy baseline), the
`runs/*/baselines.jsonl` (the DoNothing anchor values), the `RedispReward` source,
and `PLAN-BENCH.md`. One episode (chronic 500) matched do-nothing to **exactly**
0.0000 improvement after reading that DN scores 6700 there — i.e. it read the target
and matched it, not operated. **These numbers do not measure grid operation.**
(The baselines ladder — DoNothing=0, Random=−0.97, expert≈0 — is still valid, since
those run with no model/LLM involved.)

**Fix (validated):** every benchmark LLM episode now runs opencode inside a
**bubblewrap filesystem sandbox** (`bench/sandbox.py`, `OPENCODE_SANDBOX=1`). The
model sees only the operator workspace (`cli/simctl` + `AGENTS.md` + `docs/` +
`recipes/` + `render/`) bind-mounted at the repo path; `bench/`, `runs/`,
`contracts/`, `backend/` do not exist in its filesystem (verified: `cat bench/
score.py`, `ls bench/`, `ls runs/` → "No such file or directory"; `ls /home/nate/`
shows only the clean workspace). Each sandbox gets an isolated per-port HOME (own
opencode config+auth) so parallel episodes don't collide; opencode runs as root
under `sudo bwrap` (user ns disabled) so cleanup uses `sudo kill`.

A **valid** first read comes from the sandboxed runs (a smoke episode is running
now to validate the full driver+sandbox path before the standard run).

**Directional hints from the (invalidated) leaky pilot** — not a clean measurement,
but suggestive: the model *could* operate safely (0 trips across all 6) and, when
it chose to act, used **redispatch** (2–4 acts/ep, no topology changes) — the
correct loss-reduction lever on the easy horizon. **The 3–4× token inflation on
the later episodes (2.6–3.9M vs 0.8M) is now explained**: it spent tens of minutes
reading the harness's own source (`bench/`, `backend/`, `contracts/`) instead of
operating — behavior the sandbox removes entirely. So valid episodes should be
*both* cheaper (no codebase to explore) and cleaner (no answer key to read).

**Reading (provisional, to be confirmed by sandboxed runs):** on the @96 easy
horizon the LLM is likely a *safe operator that does not beat passive on reward*
— the sandbox is easy (nothing trips) and redispatch's cost (α=5.0) can exceed the
loss savings at calm operating points. The discriminating test is **@1200
(standard)**, where DN dies (5–14% survival) and the model must (a) outlast
passive and (b) find redispatch headroom under real stress.

**Cost:** ~0.8M input tokens / 96-step episode when operating (context re-sent per
turn; opencode auto-compacts as it fills the 1M window), ~20–40 min wall.
Sandboxed episodes should run at the lower end (no exploration detour).

---

## 10. Second robustness bug found & fixed: dead agents under concurrency (2026-10-04)

Running the pilot @96 and standard @1200 **simultaneously = 5 concurrent sandboxed
opencodes** (plus baselines) exposed a second robustness hole: the driver gave up on
opencode startup after only **15 s**, and under 5-way load the sandboxes came up
slower, so the agent silently died ("steer ignored: agent not running" ×N, zero
streaming) **yet the sim kept auto-advancing** and the episode was scored as a fake
LLM result (e.g. chronic 500 @1200 "completed" at a t=480 game-over after burning
2.4 h with 0 model activity; the whole @96 pilot sat at t=0 for 3 h).

**Fix (all validated):**
- opencode startup window 15 s → **180 s with retries**, stderr now logged
  (`runs/opencode-<port>.log`) for diagnosis.
- **Fail-fast liveness gate** (`--liveness-min`, default 20): if the agent shows zero
  activity that long, the episode aborts as `agent_failed` instead of burning hours.
- `agent_failed` episodes are **excluded from scoring** and listed under "INFRA
  FAILURES" in the report.
- **Concurrency capped at 3** (5 is the observed failure threshold).
- `bench_watch` prefers the newest non-`agent_failed` result per chronic so a stale
  episode can't shadow a fresh one.
- The dead-agent episodes are quarantined to `runs/_invalidated_agent_failed/`.

### 10b. Third robustness bug: mid-run WEDGES + orphaned sandboxes (fixed)

"Unlimited thinking" has a dark side the pilot didn't expose at scale: after a
runaway turn (chronic 500 burned **25M tokens** over 5 steps), the opencode *session*
went **quiet forever** (no deltas) and the episode sat at t=5 for 10 h — my liveness
counter was fooled by idle/`turn_end` events. Separately, killing the bwrap wrapper
**orphaned the inner `opencode.exe`** at 100% CPU (it's in its own pid/mount ns).

**Fixes (all validated):**
- **Stall watchdog** (`run_llm.py --stall-min`, default 40): if the model produces no
  real output (deltas/tool) that long, the backend `restart_agent` aborts the wedged
  session and creates a **fresh** one (a wedged opencode session stops generating even
  on new prompts — only a new session recovers). If it wedges again → `agent_failed`.
- Liveness now tracks **real model output** (`last_model_output_ts`), not idle events.
- **Sandbox reaping** (`sandbox.kill_sandbox_proc` + per-port ws): kills the bwrap
  wrappers **and** the namespaced `opencode.exe` (found by unique port), and `run_llm`
  does a port sweep after backend shutdown. No more orphaned 100%-CPU processes.
- **Per-port workspace** (`.sandbox-ws-<port>`) — a shared ws path that got rmtree'd on
  every `build_ws` call would wipe a parallel sibling's workspace mid-run.
- **Validated:** the exact chronic-500 that wedged 10 h now ran to t=514 in **63 min**
  with continuous output (no wedge, `agent_failed=false`); orphan reaping confirmed
  (0 root opencode/bwrap left after kill).

### 10c. Concurrency-induced gateway wedge (finding + operating point)

Panel top-up (chronics 1, 450, 700 launched at conc 3): **700 completed fine, but 1
and 450 wedged** — the opencode session came up `active` but the model returned **empty
completions** (0 tokens) even for a trivial "OK" prompt, and even after the watchdog's
fresh-session restart. Direct gateway calls worked (5/5 "OK"), and the per-port auth/config
were identical to the working episode — so it was a **transient gateway hiccup hit by the
two sessions that started simultaneously with a third** (3 heavy reasoning sessions at once).

**Operating point: conc 2 is reliable; conc 3 wedges ~2/3.** Re-launched chronic 1 and 450
at conc 2 → both generating immediately. Run the panel at conc 2, and if a session wedges
the watchdog marks it `agent_failed` + we restart at conc 1.

## 11. First VALID @1200 read (4 episodes, all DN-death chronics) — 2026-10-04

| chronic | LLM survived | DN died @ | trips | cum (LLM vs DN) | taxonomy |
|---|---|---|---|---|---|
| 0 | **1200/1200** | 1091 | 0 | 73597 vs 69237 (+6%) | **OUTLASTED-DN (win)** — surgical `change_bus` on 12_13_14 at the DN-cascade point |
| 200 | 377 | 486 | 0 | 26177 vs 33548 | died-earlier-than-DN \| **loss-of-load/non-trip** — moved ALL loads at sub 12 to bus 2 |
| 500 | 514 | 514 | 5 | 29330 vs 34538 | tied-DN-death \| trip-cascade (5 trips) |
| 900 | 381 | 381 | 5 | 26153 vs 26299 | tied-DN-death \| trip-cascade (5 trips) |

Pattern so far: **1 win / 1 loss / 2 ties** — roughly do-nothing-equivalent. The win
(chronic 0) was a *targeted* re-route at exactly DN's cascade point; the loss (200) was
a specific anti-pattern (loss-of-load); the ties died on trip-cascades at ~DN's step.
Cost is high and variable (chronic 0 ≈ 29M input tokens / 9 h; chronic 500 ≈ 6.5M / 63
min). Panel top-up to n=7 (chronics 1, 450, 700) is running conc 3 with the fixed
harness.

**Key actionable finding:** chronic 200's failure is a specific, learnable
anti-pattern — the model re-busbar'd **every load at a substation** to bus 2 in one
burst, dropping the load (loss-of-load game-over, 0 trips). A targeted docs/pitfalls
warning ("never move all loads/gens at a substation to the other busbar at once — it
detaches them") is the highest-value single harness intervention to test next
(kept as a *separate config*, not a baseline change, for comparability).

## 12. Roadmap (what's next)

1. **Complete the @1200 panel to n=7** (chronics 1, 450, 700 running conc 3 now).
   Establishes whether "1 win / 1 loss / 2 ties" is the real pattern or noise, with a
   real CI. *(in progress)*
2. **Loss-of-load A/B** — `bench/configs/baseline_warn_loads.json` is **built +
   validated** (appends a targeted warning to AGENTS.md; per-port ws; everything else
   = baseline). Run chronic 200 (the loss) with the warning vs the baseline result we
   already have. Highest-value single intervention (R2 "harness-as-variable").
   *(ready to launch as a slot frees)*
3. **Valid @96 pilot** (easy horizon, fast/cheap) with the fixed harness — the "safe on
   easy grids" read.
4. **Ablations** on a couple of @1200 chronics: `no_docs` (can it discover the API
   without our docs?), `no_vision` (is the image channel worth ~40 s/turn?),
   `detailed_obs` (verbosity cost/benefit).
5. **Difficulty extension (Phase 3):** `l2rpn_wcci_2020` (36-sub; data not yet
   downloaded) — survival becomes the differentiator where a real skill edge would
   show.
6. **Adversarial (Phase 3):** grid2op native `Opponent` (robustness) → 2-agent
   defender-vs-attacker (ELO/Bradley-Terry; no judge needed, ground-truth in sim).

Note on comparability: the sandbox **copies** `docs/`+`AGENTS.md`+`recipes/` into each
episode's workspace at start, so every episode is frozen to the docs-at-that-moment.
Editing the baseline docs mid-benchmark would create a *new* config — so doc changes
are run as separate configs (ablations), never a silent baseline edit.

---

## 13. Session results: cross-model + adversarial mode (2026-10-04)

### 13.1 Final @1200 panel (n=7, AA-Dense-Blackwell, all DN-death chronics)
0=W, 1=W, 200=L, 450=W, 500=T, 700=W, 900=T → **4 better / 2 tied / 1 worse**,
mean **+12.1% over do-nothing** (CI [−8%, +32%]), survival 4/7 vs DN 0/7.
The wins are *targeted* re-routes at DN's cascade points; the loss (200) is the
loss-of-load anti-pattern (moved all loads at a sub to bus 2). Cost ~3.5–31.7M
input tokens/episode (DN is free).

### 13.2 Cross-model (AA-General vs AA-Dense-Blackwell, same @1200 panel)
**Mechanism:** opencode's session model comes from the config's `agent.build.model`
pin, NOT a session override — so the harness rewrites the per-sandbox `opencode.json`
(`bench/sandbox.py build_sandbox_home(model=)`). Verified: `--model AA-General`
actually runs AA-General.
**Result (chronics 0 & 200):** AA-General is ~**8× cheaper** (3–4M vs 24–29M tokens,
~8min vs ~100min) but **splits the panel** — it *loses* to AAB on chronic 0 (self-
inflicted blackout at t=1087 vs AAB's 1200 win) and *beats* AAB decisively on
chronic 200 (480 vs 377 survived). Neither was a net win over DN on those two.
Takeaway: the models are close in skill, hugely different in cost; the panel (more
chronics × both models) is the clean comparison.

### 13.3 Adversarial / robustness mode (the "sick" feature — built + validating)
**Design:** LLM defender vs a deterministic SCRIPTED attacker (grid2op's native
opponents are unreliable in 1.12 — sparse/buggy). The attacker is a seeded schedule
of "cut line L for d steps" fired by the BACKEND at exact sim-steps (pace-independent,
so the LLM and the do-nothing anchor see *identical* attacks). The defender is blind
to who attacked — it only sees line failures. **Fair anchor** = do-nothing under the
same attacks (computed free, in-process).
- `bench/attacker.py` — deterministic schedule (verified reproducible: same seed →
  same deaths). DN dies @1091 natural but @374 under attack (attack interval 96,
  dur 24) — a ~2.5× threat.
- `backend` — `SimSession.set_attack_schedule` + `_fire_attacks` (fires at every step
  boundary), `/sim/attack` (source=opponent, emits `opponent.action`/`opponent.step`),
  `/bench/start` accepts `attacks`. **Defender-blindness verified** (last_action.source=
  opponent, model told nothing). Backend 25/25.
- **Web UI human-attack panel** (subagent-built, verified: build clean, 26/26 tests) —
  the user cuts/re-closes lines via `POST /sim/attack`; the feed shows red `[ATTACK]`
  rows; the defender is never told.
- **First adversarial LLM episode RUNNING** (AAB defender vs scripted attacker,
  chronic 0 @1200, DN-under-attack anchor = 374). Question: does AAB still win under
  attack, or do the attacks push it below do-nothing?
- **Agent-vs-agent (building):** LLM defender + LLM attacker on one shared grid, each
  blind to the other's reasoning (attacker uses `simctl attack`, defender `simctl act`).

### 13.4 More robustness fixes (found via the cross-model probe)
- **Terminal-state corruption:** the runner re-fetched `/bench/stats` *after* the
  episode ended; if the backend crashed in that window it read a fresh sim (t=0) and
  saved a bogus record. Fixed: build the result from the CAPTURED terminal state (a
  defensive re-fetch is only used if it still shows the same terminal episode).
- **Concurrency:** 3 concurrent heavy reasoning sessions wedged 2/3 (gateway returned
  empty completions). **Operating point = conc 2.** The stall watchdog + fail-fast now
  catch a wedge that conc-3 produces.

### 13.5 Harder-grid data — the honest blocker
`grid2op.make(<harder grid>)` (the docs' recommendation) auto-downloads, but the
official mirrors are dead: the sandbox is on HuggingFace (`grid2op/l2rpn-datasets`,
sandbox-only), the harder grids point to an Azure blob (`l2rpnukstorageprem...`) that
is **403 "account disabled" on the whole bucket**, and the rest are CodaLab-login-gated.
So the only *free, unblocked* grid is the case14 sandbox. To unlock wcci_2020 /
neurips_2020_track1 (the docs-recommended robustness grids) we need a **CodaLab
account** (accept terms → download the kit → drop it at `~/data_grid2op/<env>/`), after
which `grid2op.make(...)` finds it locally with no download. Everything else in the
benchmark (adversarial, cross-model, web UI) runs fine on the sandbox meanwhile.

### 13.6 Updated roadmap
1. **Finish the adversarial read** (AAB-under-attack chronic 0, running) — does the
   defender hold up vs do-nothing-under-same-attacks? Then a few more chronics.
2. **Agent-vs-agent** (building) — defender vs attacker LLMs, scored by survival /
   time-to-blackout; the user's flagship feature.
3. **Full cross-model panel** — AAB + AA-General (+ AA-Fast-Vision) on the same @1200
   panel → a real multi-model leaderboard (skill vs cost).
4. **Harder grid** — once CodaLab data is available (wcci_2020 / neurips_2020_t1).
5. (Deprioritized per user: the single loss-of-load docs A/B.)
