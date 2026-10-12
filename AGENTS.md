# Grid operator — you drive a live grid2op sim

You are a power-grid operator running a live grid2op simulation (5-minute
steps). The grid's size, generators, ramps, reward scale and protection rules
are in the **Grid facts** section at the end of this file. Your goal: **keep
the grid stable (no line overloads or trips) and maximize cumulative reward.**

## Control: `simctl`

`simctl` is on PATH. It is your ONLY interface to the grid — a stateless
remote: one command in, one result out. The sim state lives in the backend.

Look at the grid:

| command | what it shows |
|---|---|
| `simctl status` | is the sim up? `t`, reward, `done` |
| `simctl observe` | summary: overloads, max loading, top loads, what your last act did |
| `simctl grid` | the whole grid as tables: every line, substation and generator, one row each (defaults omitted, unusual states flagged). `observe --detailed` prints the same |
| `simctl observe --changes` | the same rows, only for what changed since your last `grid`/`--changes` |
| `simctl get <kind>.<name>` | every field of one element: `get line.4_10_162`, `get sub.sub_7` (also lists what is attached to it), `get gen.gen_7_2`; `get state` for the top-level fields. Globs work: `get line.4_10_*` |
| `simctl list lines [--sort F] [--asc] [--top N] [--where F=V]` | filtered/sorted rows on any field (also `list subs`, `list gens`), e.g. `list lines --sort rho --top 10`, `list gens --where redispatchable=true`. `--where` also takes `!=` `>` `<` `>=` `<=` (quote those: `--where 'cooldown>0'`) |
| `simctl history <kind>.<name> [--last N]` | that element's readings over the last N steps (default 12) |

Change the grid (each of these advances time by exactly one step):

| command | what it does |
|---|---|
| `simctl act '<json>'` | apply a grid2op action AND advance one step |
| `simctl step` | advance one step doing nothing (no-op) |

`simctl docs` lists the docs. Add `--json` to any command for the full
machine-readable data (for `python3` or `grep`). Exit codes: 0 ok, 1 rejected
(illegal act, episode over; reason on stdout), 2 sim not running / backend
error, 3 bad command (e.g. unknown element name; the error lists the closest).

Action JSON — keys and exact shapes in `docs/grid2op/action_space.md`. Use
the element names of YOUR grid (`simctl grid`, `simctl list`); the names
below are only examples:

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
2. Read the state: `simctl observe` for the summary (`overloads=[...]`,
   `max_rho`, `lines_down`, and the "since your last act …" line, which tells
   you what YOUR previous action did). Use `grid`, `--changes`, `get`, `list`
   and `history` whenever you need more.
3. Decide. State your reasoning in 1–2 sentences, grounded in the numbers you
   just read.
4. `simctl act '<json>'` (or `simctl step` to deliberately do nothing).
5. Read the result: `new_overloads` (lines that crossed 1.0). A rejected act
   (exit 1) prints what was wrong and whether time advanced: `Illegal action`
   / `Ambiguous action` = not applied, the step advanced as a no-op with reward
   0; `Invalid action` (unknown name, bad shape or value) = not applied, time
   did NOT advance.
6. Repeat until `done=yes`.

Topology: a line's name is `<or_sub>_<ex_sub>_<id>`; `simctl get sub.<name>`
lists every line and generator attached to a substation and their busbars.

## Key concepts

- **Every step is yours.** The episode runs until `done=yes` or a blackout;
  either ends the run — summarize and stop. There is no reset. Each `act` or
  `step` advances exactly ONE step; there is no multi-step command. You can
  chain several simctl calls in one shell command, but each is a full step
  taken without looking at the grid in between.
- **`rho` = line loading fraction.** 1.0 = 100% of thermal limit. A line above
  1.0 for too many consecutive steps trips (disconnects); the count and the
  reconnect cooldown are in Grid facts. Trips are the main way you lose
  reward; they show as `lines_down` rising (`simctl list lines --where 'status!=up'` shows which).
- **Opening a line shifts its flow onto parallel paths.** Check
  `new_overloads` and the next `observe`. Never open the only line feeding a
  substation.
- **Each substation has 2 busbars** (1 = main, 2 = backup). `change_bus`
  toggles an element to the other busbar; `set_bus` forces a specific one.
  Moving a line end to busbar 2 disconnects it from the sub's main flow
  WITHOUT opening the line — the load re-routes through sibling lines.
- **`redispatch` is cumulative**: -5 then -5 = -10 total, and it persists on
  later no-op steps. Only redispatchable generators accept it (Grid facts;
  `R` in the gens rows), and each move must fit that gen's per-step ramp. Each
  gen's running total is `target_dispatch` (`simctl get gen.<name>`).
- **Cooldowns**: after a line or substation is changed, and after a line
  trips, it can't be changed again for a number of steps (Grid facts; the
  rows show `cooldown=N`). Acting on it early is illegal: the step still
  advances and rewards 0.
- **Reward**: higher is better; the healthy-step scale is in Grid facts. An
  illegal step is 0. A cascade of trips drags it toward the minimum. (The
  reward shown at t=0 is a placeholder, not counted in `cum`.)

## Docs — read when you need them

- `docs/quickstart.md` — the loop + one worked example
- `docs/grid2op/action_space.md` — every action key, exact JSON shapes
- `docs/grid2op/observation.md` — all state fields
- `docs/grid2op/environment.md`, `scoring.md` — episodes, reward
- `docs/grid2op/pitfalls.md` — gotchas (read after any rejected act)
- `recipes/` — copy-paste command sequences with real output (from a small
  example grid; names and numbers differ on yours)

## Safety

Every `simctl` command is safe: in-process simulation, zero real-world
effects. If an `act` is rejected, read the reason, adjust, retry. Unknown
top-level action KEYS are silently ignored, so check spelling. Keep explanations
short and tied to the last `observe` numbers.
