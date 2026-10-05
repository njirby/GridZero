# Observation — reading the grid state

Two levels: `simctl observe` (compact text, what you read every turn) and
`simctl observe --detailed` (full JSON — the C3 GRID-STATE shape, one object
per line/sub/gen). The JSON below is a real snapshot at t=50 of
`l2rpn_case14_sandbox` (from `contracts/examples/grid-state-t50.json`).

## Compact `observe` — the five lines

```
t=52 reward=61.2 (cum 3074.8) done=no
lines_down=1  max_rho=0.97 (2_3_5)  overloads=[2_3_5]
top loads:  2_3_5=97.0%  0_4_1=88.0%  5_12_9=71.0%
gens: 81.4 79.3 5.3 0.0 | loads: 5.4 12.6 14.4 ...
since your last act (t=51 set_line_status 0_4_1=-1): 2_3_5 +9.0%, no trips
```

- Line 1: clock, step reward, cumulative reward, episode state.
- Line 2: count of down lines, the hottest line (name at its rho), and all
  lines above 1.0 right now.
- Line 3: the 3–4 most loaded lines, percent of limit.
- Line 4: generator MW (order = `env.name_gen`) and top load MWs.
- Line 5: **feedback on your last act** — which lines moved because of it,
  and whether anything tripped since. This is your steering signal; read it
  before deciding.

## The JSON (`--detailed`) — top level

| field | meaning |
|---|---|
| `t`, `max_t` | step 50 of 8064 (5-min steps, 28 days) |
| `reward` | this step's reward (69.5 here) |
| `cum_reward` | episode total so far (3323.5 here) |
| `done`, `cause` | episode over? why (null = still running) |
| `delta_min` | 5.0 — minutes per step |
| `sim_clock` | `2019-01-06 04:10:00` — the grid's wall clock |
| `n_line`/`n_sub`/`n_gen` | 20 / 14 / 6 (loads: 11) |
| `max_rho` | hottest line's loading (1.1737 here — OVERLOADED) |
| `n_down` | lines currently disconnected (1 here) |
| `n_overflow` | lines above 1.0 right now (1 here) |
| `lines` | one object per line, in id order (see below) |
| `subs` | one object per substation, with (x, y) layout coords |
| `gens` | one object per generator |
| `alarms` | raised alarms (empty here) |
| `last_action` | the previous act: source, summary, args, t |
| `png` | path of the latest render, if any |

## Per-line object

```json
{
  "id": 4, "name": "1_4_4", "or": "sub_1", "ex": "sub_4",
  "rho": 1.1737, "p_or": 50.537, "p_ex": -49.165,
  "status": "up", "overflow": true, "cooldown": 0, "maint": -1
}
```

- `name` — `<or_sub>_<ex_sub>_<id>`. `or`/`ex` name the substations at each
  end. This is how you learn topology from the JSON (or just `render`).
- **`rho` — the number to watch.** Fraction of the line's thermal limit,
  computed from CURRENT (amps), not from `p_or`: `rho = max(|a_or|,|a_ex|) /
  limit_amps`. **>1.0 = overloaded.** Two consecutive steps >1.0 → the line
  trips. Keep `max_rho` under ~0.9 for comfort. Don't try to recompute rho
  from `p_or` — the limit is in amps and flows have reactive power, the math
  won't match.
- `p_or`, `p_ex` — active power (MW) entering at the or end and the ex end.
  Flow or→ex reads `p_or > 0`, `p_ex < 0` (sign convention). Magnitudes
  differ slightly (line losses).
- `status` — `up`, `down`, `cooldown`, or `maintenance`. A down line has
  `rho: 0.0` and `p_or: 0.0`.
- `overflow` — currently above 1.0 (bool). `cooldown` — steps left before
  the line can be re-closed (0 = not in cooldown). `maint` — steps until the
  next scheduled maintenance, `-1` = none (the sandbox has no maintenance).
- In the t=50 snapshot: `0_4_1` is `down` (opened by the operator at t=49,
  `rho: 0.0`), and its sibling `1_4_4` is at `rho: 1.1737, overflow: true` —
  one more hot step and 1_4_4 trips. That's the pattern to catch early.

## Per-sub object

```json
{ "id": 4, "name": "sub_4", "x": -64.0, "y": -54.0, "type": "load", "p": -6.3 }
```
`type`: `gen` (sub_0, sub_7), `load` (pure consumption), `both` (sub_1,
sub_2, sub_5 — gen and load on the same sub), `other`. `p` is net MW at the
sub (negative = net import). `x`/`y` are layout coordinates — same numbers
`render` uses.

## Per-gen object

```json
{ "id": 0, "name": "gen_1_0", "sub": "sub_1", "p": 75.0,
  "renewable": false, "redispatchable": true }
```
`redispatchable: true` = you can `redispatch` it (gen_1_0, gen_2_1, gen_0_5).
Renewables (gen_5_2, gen_5_3, gen_7_4) can't be redispatched but CAN be
curtailed (`curtail`), and their `p` moves with the wind/solar chronic.

## Advanced `obs` attributes (what the JSON fields come from)

If you want to go deeper than the C3 JSON, these are the raw grid2op
`Observation` attributes the backend reads (verified in 1.12.5):

- `obs.rho` — float array [20], loading per line (same as `lines[].rho`)
- `obs.line_status` — **bool** array, `True` = connected (C3 renders it as
  the `status` string)
- `obs.gen_p` / `obs.load_p` — MW per gen / per load
- `obs.gen_margin_up` / `obs.gen_margin_down` — how far you may
  `redispatch` each gen THIS step without the action going ambiguous
  (t=0: gen_1_0 ±5, gen_2_1 ±10, gen_0_5 ±15, renewables 0)
- `obs.current_step` / `obs.max_step` — t / 8064
- `obs.grid_layout` — `{sub_name: (x, y)}`, what `render` draws from
- `obs.get_time_stamp()` — the sim clock as a datetime
- `obs.timestep_overflow` — per line, consecutive steps already spent >1.0
  (1 means "trips next step if still hot")
- `obs.time_before_cooldown_line` — per line, steps left of reconnection
  cooldown (≈10 right after a trip; 0 = free)
- `obs.time_next_maintenance` — per line, steps to next maintenance (-1 none)
- `obs.simulate(action)` — **free dry-run**: feeds an action through the
  forecast and returns `(next_obs, reward, done, info)` WITHOUT advancing the
  real grid. The backend uses it: when it predicts a trip from your act,
  `predicted_disc_lines` shows up in `act --json`. You can ask for the
  prediction by reading `new_overloads`/`predicted_disc_lines` in the act
  result instead of gambling.

You don't call these directly (the backend owns the env) — they explain
where every JSON field comes from and what the sim knows.
