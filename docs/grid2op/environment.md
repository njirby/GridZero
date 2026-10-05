# The environment: episodes, steps, time

Everything below is verified against grid2op 1.12.5 running
`l2rpn_case14_sandbox`.

## Episode model

An **episode** is one run of the grid from a fixed start until it ends.

- `simctl reset` → new episode at `t=0`. Grid starts with all 20 lines
  connected, generators at base setpoints (gen_p: 81.4, 79.3, 5.3, 0.0, 0.0,
  82.2 MW).
- Each `simctl act` or `simctl step` advances **one 5-minute timestep** and
  returns the new state. `simctl step N` does the same N times with
  do-nothing actions (watch natural load variation).
- `t` counts 0 → **8064** max (`obs.max_step`), i.e. 28 simulated days.
  The sim clock starts 2019-01-04 00:00; at t=50 it reads 2019-01-06 04:10.
- `done=true` ends the episode. Causes: a cascading failure (too many lines
  down / subs isolated), or t reaches 8064. **After `done`, the observation
  is a "game over" state — don't read it, call `simctl reset`.**

## Step semantics (from `Environment.step`'s docstring)

`step(action)` returns `(observation, reward, done, info)`.

> "If the BaseAction is illegal or ambiguous, the step is performed, but the
> action is replaced with a 'do nothing' action."

That means: **a rejected act does not freeze time.** The grid steps forward,
your move is dropped, `info.is_illegal` (or `is_ambiguous`) is true, and the
step's reward is 0. You still need to act next turn.

`info` keys you'll see via `simctl act --json`:

| key | meaning |
|---|---|
| `is_illegal` | action was not allowed (e.g. closing a line in cooldown) |
| `is_ambiguous` | action's effect can't be determined (e.g. redispatch beyond a gen's margin) |
| `disc_lines` | per-line: -1 = not disconnected this step, 0 = disconnected this step (cascade cause), 1,2,… = disconnected later in the cascade |
| `failed_redispatching` | redispatch part was infeasible / ignored |
| `opponent_attack_line` / `opponent_attack_sub` / `opponent_attack_duration` | v1+: adversarial attacks, if any |

## Load and generation follow the data (chronics)

The sandbox replays a recorded **chronic**: 5-minute load and renewable-output
time series. You don't set loads; they drift a few MW per hour. Renewables
(gen_5_2, gen_5_3, gen_7_4) vary with the series — that's what moves
`rho` when you do nothing. `simctl step` is how you watch that drift.

A **scenario** is one draw from the data set (start timestamp, load curve,
attack schedule); each `reset` in the backend uses the configured scenario.
You don't manage scenarios — `simctl reset` is all you have.

## Grid shape (case14 sandbox)

- 14 substations (`sub_0`…`sub_13`), each with **2 busbars**.
- 20 power lines named `<or_sub>_<ex_sub>_<idx>` (e.g. `0_4_1` = sub_0→sub_4,
  id 1). `name` order == id order: `0_1_0` is id 0, `6_8_19` is id 19.
- 6 generators: 3 dispatchable (`gen_1_0` sub_1, `gen_2_1` sub_2, `gen_0_5`
  sub_0) and 3 renewables (`gen_5_2`, `gen_5_3` sub_5, `gen_7_4` sub_7).
- 11 loads.
- Thermal limits via `env.get_thermal_limit()` — **in AMPS** (541, 450, 375,
  636, 175, 285, … for the first six lines), not MW.
- Step cost ~58 ms; render ~750 ms. No need to be frugal with `observe`,
  but `render` is for when you actually want to look.

## What you can NOT do

- No grid object on the env in grid2op 1.12 (the old attribute is gone) —
  never write code assuming it; you drive the grid via `simctl` anyway.
- No direct python access to the env — the backend owns it, single writer.
- No storage units in this grid (`set_storage` raises).
- No detach/attach of loads or gens in this env's action space (the keys are
  silently ignored — see `pitfalls.md`).
