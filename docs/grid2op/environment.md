# The environment: episodes, steps, time

Everything below is verified against grid2op 1.12.4 running
`l2rpn_case14_sandbox`.

## Episode model

An **episode** is one run of the grid from a fixed start until it ends. The
backend starts it for you; **you cannot reset, restart or fast-forward it.**

- At `t=0` all lines are connected and generators sit at the scenario's base
  setpoints. `simctl status` shows the horizon as `t=0/<max_t>` (training
  tasks use a short horizon such as 24; the full sandbox episode is 8064
  steps = 28 days).
- Each `simctl act` or `simctl step` advances **one 5-minute timestep** and
  returns the new state. `simctl step` is a do-nothing step; `simctl step 3`
  is refused (exit 3) — run `simctl step` three times.
- `reward` at `t=0` (`-10.0`) is a placeholder from the reset, not a step
  reward; `cum` only sums completed steps.
- `done=yes` ends the episode. Causes: `t` reached `max_t` (`time_exceeded`),
  or a blackout / cascading failure (`game_over`, `observe --detailed` has
  the `cause`). **After `done` the episode is over: `act` and `step` print
  `Episode is over (<cause>). Stop acting and write your summary.` and exit
  1. Stop and summarize.** `status`/`observe` still show the final state.

## Step semantics (from `Environment.step`'s docstring)

`step(action)` returns `(observation, reward, done, info)`.

> "If the BaseAction is illegal or ambiguous, the step is performed, but the
> action is replaced with a 'do nothing' action."

That means: **a rejected act does not freeze time.** The grid steps forward,
your move is dropped, and the step's reward is 0. Exception: an action that
cannot even be built (wrong JSON shape, unknown line/gen name, bus 3) prints
`Illegal — <reason>` (exit 1) and does NOT advance time.

What `simctl act --json` returns (the C2 envelope):

```
{"ok": true|false, "error": null|"<reason>",
 "data": {"t", "reward", "cum_reward", "done", "lines_down", "overloads": [...],
          "disc_lines": [...], "new_overloads": [...], "illegal", "ambiguous",
          "applied": {...}},
 "verbose": {"is_illegal", "is_ambiguous", "opponent_attack_line", "predicted_disc_lines"}}
```

| key | meaning |
|---|---|
| `ok` / `error` | `false` + `Illegal action: ...` / `Ambiguous action: ...` when the sim rejected it (exit 1) |
| `new_overloads` | lines that crossed 1.0 on this step |
| `overloads`, `lines_down` | all lines above 1.0 / currently disconnected, after the step |
| `disc_lines` | lines that tripped on this step (see `observe --detailed` `status: down` to confirm) |
| `illegal` / `ambiguous` | the action was dropped (reward 0, clock advanced) |
| `verbose.predicted_disc_lines` | lines the backend's dry-run predicted would trip from your move |

## Load and generation follow the data (chronics)

The sandbox replays a recorded **chronic**: 5-minute load and renewable-output
time series. You don't set loads; they drift a few MW per hour. Renewables
(gen_5_2, gen_5_3, gen_7_4) vary with the series — that's what moves
`rho` when you do nothing. `simctl step` is how you watch that drift.

A **scenario** is one draw from the data set (start timestamp, load curve).
The operator picks it; you don't manage scenarios, and numbers in the docs
(loadings, MW) differ from scenario to scenario.

## Grid shape (case14 sandbox)

- 14 substations (`sub_0`…`sub_13`), each with **2 busbars**.
- 20 power lines named `<or_sub>_<ex_sub>_<idx>` (e.g. `0_4_1` = sub_0→sub_4,
  id 1). `name` order == id order: `0_1_0` is id 0, `6_8_19` is id 19.
- 6 generators: 3 dispatchable (`gen_1_0` sub_1, `gen_2_1` sub_2, `gen_0_5`
  sub_0) and 3 renewables (`gen_5_2`, `gen_5_3` sub_5, `gen_7_4` sub_7).
- 11 loads.
- Thermal limits via `env.get_thermal_limit()` — **in AMPS** (541, 450, 375,
  636, 175, 285, … for the first six lines), not MW.
- `observe` is cheap; `render` (when enabled) is for when you actually want
  to look.

## What you can NOT do

- No grid object on the env in grid2op 1.12 (the old attribute is gone) —
  never write code assuming it; you drive the grid via `simctl` anyway.
- No reset: don't call `simctl reset` or the backend's `/sim/reset`; they are
  not part of your interface.
- No direct python access to the env — the backend owns it, single writer.
- No storage units in this grid (`set_storage` raises).
- No detach/attach of loads or gens in this env's action space (the keys are
  silently ignored — see `pitfalls.md`).
