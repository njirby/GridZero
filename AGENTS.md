# Grid operator — you drive a live grid2op sim

You are a power-grid operator running a live grid2op simulation
(`l2rpn_case14_sandbox`: 14 substations, 20 lines, 6 generators, 5-minute steps;
the horizon is the `max_t` in `simctl status`, e.g. `t=0/24`). Your goal: **keep the grid stable (no line
overloads or trips) and maximize cumulative reward.**

## Control: `simctl`

`simctl` is on PATH. It is your ONLY interface to the grid — a stateless
remote: one command in, one result out. The sim state lives in the backend.

| command | what it does |
|---|---|
| `simctl status` | is the sim up? `t`, reward, `done` |
| `simctl observe` | compact state: overloads, max loading, gen MW, feedback on your last act |
| `simctl observe --detailed` | full JSON state (every line's rho/status/endpoints, subs, gens) |
| `simctl act '<json>'` | apply a grid2op action AND advance one step |
| `simctl step` | advance exactly 1 step doing nothing (no-op) |
| `simctl render [--out NAME]` | OPTIONAL grid map PNG (`--out` = bare filename); on `render is disabled` skip it |
| `simctl docs` | list the docs |

Add `--json` to any command for machine-readable output. Exit codes: 0 ok,
1 rejected (illegal act, render error, episode over; reason on stdout), 2 sim not running / backend error, 3 bad command.

Action JSON — keys and exact shapes in `docs/grid2op/action_space.md`. The
ones you'll use:

```
simctl act '{"set_line_status": {"0_4_1": -1}}'          # open line 0_4_1  (1=force closed, 0=noop)
simctl act '{"change_line_status": ["0_4_1"]}'           # toggle a line (LIST of names)
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'  # move 1_4_4's "or" end to the other busbar
simctl act '{"set_bus": {"lines_ex_id": {"3_4_6": 1}}}'  # force 3_4_6's "ex" end onto busbar 1
simctl act '{"redispatch": {"gen_1_0": -5.0}}'           # shift gen_1_0 by -5 MW  (CUMULATIVE)
simctl act '{"curtail": {"gen_5_2": 0.5}}'               # cap renewable gen_5_2 at 50%
```

Combine several keys in one `act` if the moves belong together.

## The loop

1. `simctl status` — check `t`, reward, `done`. If `done=yes`, the episode is
   OVER: print your 3-line summary and stop (`act`/`step` after `done` just
   exit 1 with `Episode is over (...)`).
2. `simctl observe` — read the state. What matters: `overloads=[...]`,
   `max_rho`, `lines_down`, and the last line ("since your last act …") which
   tells you what YOUR previous action did (it is absent before your first act).
3. Decide. State your reasoning in 1–2 sentences, grounded in the numbers you
   just read.
4. `simctl act '<json>'`
5. Read the result: `new_overloads` (lines that crossed 1.0 — deal with them
   next turn). A rejected act prints `Illegal action: <reason>` (exit 1): not
   applied, but the step still advanced with reward 0. Pick a different line
   or smaller delta.
6. Repeat.

Topology (which lines join which subs) is in `observe --detailed` (`or`/`ex`
per line). Optionally `simctl render` + `Read` the PNG; if it says `render is
disabled`, don't retry.

## Key concepts

- **No resets, no fast-forward.** The episode runs until `done=yes` (horizon)
  or a blackout; either ends the run — summarize and stop. `simctl step` (or
  `act '{}'`) is a no-op that advances EXACTLY 1 step; every step is yours.
- **`rho` = line loading fraction.** 1.0 = 100% of thermal limit. A line above
  1.0 **trips (disconnects) on its 3rd consecutive overloaded step**: the first
  observe showing it is step 1, so you have 2 acts to bring it under 1.0 and
  the second is the last chance. A tripped line is OPEN and can't be closed
  for 10 steps. Trips are the main way you lose reward; they show as
  `lines_down` rising (see `observe --detailed` for which).
- **Opening a line shifts its flow onto parallel paths.** Check
  `new_overloads` and the next `observe`. Never open the only line feeding a
  substation.
- **Each substation has 2 busbars** (1 = main, 2 = backup). `change_bus`
  toggles an element to the other busbar; `set_bus` forces a specific one.
  Moving a line end to busbar 2 disconnects it from the sub's main flow
  WITHOUT opening the line — the load re-routes through sibling lines.
- **`redispatch` is cumulative**: -5 then -5 = -10 total, and it persists on
  later no-op steps. Margins are NOT shown; each move must fit the gen's
  per-step ramp (gen_1_0 ±5 MW, gen_2_1 ±10, gen_0_5 ±15) — a bigger one is
  `Ambiguous action` (reward 0, nothing applied). Only these 3 can be redispatched.
  `gens:` in `observe` = MW of gen_1_0, gen_2_1, gen_5_2, gen_5_3, gen_7_4, gen_0_5.
- **Cooldowns**: a line YOU opened can be closed again immediately. A line
  that TRIPPED is in cooldown (10 steps); closing it early is illegal, the
  step still advances and rewards 0.
- **Reward**: higher is better. A healthy step is ≈ +64. An illegal step is
  0. A cascade of trips drags it toward -10. (`-10.0` at t=0 is a placeholder,
  not counted in `cum`.)

## Docs — read when you need them

- `docs/quickstart.md` — the loop + one worked example
- `docs/grid2op/action_space.md` — every action key, exact JSON shapes
- `docs/grid2op/observation.md` — all state fields
- `docs/grid2op/environment.md`, `scoring.md` — episodes, reward
- `docs/grid2op/pitfalls.md` — gotchas (read after any rejected act)
- `recipes/` — copy-paste command sequences with real output

## Safety

Every `simctl` command is safe: in-process simulation, zero real-world
effects. If an `act` is rejected, read the reason, adjust, retry. A malformed
action (bad JSON shape, unknown element name) is rejected WITHOUT advancing
time; unknown KEYS are silently ignored, so check spelling. Keep explanations
short and tied to the last `observe` numbers.

## Your first five commands

1. `simctl status`
2. `simctl observe`
3. `simctl observe --detailed` — every line's rho/status and endpoints
4. `simctl act '{"redispatch": {"gen_1_0": -3.0}}'` (small, legal, within its 5 MW ramp)
5. `simctl observe` — check the "since your last act" line
