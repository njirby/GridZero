# Observation — reading the grid state

Two levels: `simctl observe` (compact text, what you read every turn) and
`simctl observe --detailed` (full JSON — the C3 GRID-STATE shape, one object
per line/sub/gen). The JSON below is a real snapshot at t=50 of
`l2rpn_case14_sandbox` (from `contracts/examples/grid-state-t50.json`).

## Compact `observe` — the lines

```
t=2 reward=64.0 (cum 127.4) done=no
lines_down=1  max_rho=1.01 (1_4_4)  overloads=[1_4_4]
top loads:  1_4_4=101.2%  5_12_9=80.0%  5_10_7=65.7%
gens: 73.9 72.7 35.6 0.0 0.0 71.0
since your last act (t=0 set_line_status 0_4_1=down): no trips; overloads now: [1_4_4] (max_rho=1.01)
```

- Line 1: step, last step's reward, cumulative reward, episode state. At
  `t=0` the reward is a `-10.0` placeholder (no step has run yet).
- Line 2: count of down lines, the hottest line (name at its rho), and all
  lines above 1.0 right now.
- Line 3: the 3 most loaded LINES, percent of thermal limit (it is a line
  list, not loads).
- Line 4: generator MW in this order: gen_1_0, gen_2_1, gen_5_2, gen_5_3,
  gen_7_4, gen_0_5 (the `gens` array of `--detailed`).
- Line 5: **feedback on your last act** — `(t=<step you acted at> <summary>)`,
  then `no trips` or `tripped: [<lines>]` for the most recent step,
  `; overloads now: [...]` if any line is above 1.0, and `max_rho`. It does
  not exist before your first act (nothing printed). It names the last act
  even if that act was illegal or you have since stepped with no-ops. The
  authoritative trip check is `lines_down` plus `status: "down"` lines in
  `observe --detailed`.

## The JSON (`--detailed`) — top level

| field | meaning |
|---|---|
| `t`, `max_t` | step 50 of `max_t` (5-min steps) |
| `reward` | this step's reward (69.5 here) |
| `cum_reward` | episode total so far (3323.5 here) |
| `done`, `cause` | episode over? why: `time_exceeded` (horizon) or `game_over` (blackout); null = running |
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
| `last_action` | the previous act: source, summary, args, t (null before your first act) |
| `last_disc_lines` | lines that tripped on the latest step (`tripped: [...]` in compact `observe`) |
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
  limit_amps`. **>1.0 = overloaded.** A line trips on its 3rd consecutive step >1.0. Keep `max_rho` under ~0.9 for comfort. Don't try to recompute rho
  from `p_or` — the limit is in amps and flows have reactive power, the math
  won't match.
- `p_or`, `p_ex` — active power (MW) entering at the or end and the ex end.
  Flow or→ex reads `p_or > 0`, `p_ex < 0` (sign convention). Magnitudes
  differ slightly (line losses).
- `status` — `up`, `down`, `cooldown`, or `maintenance`. A disconnected line
  (opened by you OR tripped) is always `down`, with `rho: 0.0`, `p_or: 0.0`;
  tripped lines show their reconnection wait in `cooldown`. `cooldown` as a
  status means CONNECTED but still in cooldown after being closed.
- `overflow` — currently above 1.0 (bool). `cooldown` — steps left before
  the line can be re-closed (0 = not in cooldown). `maint` — steps until the
  next scheduled maintenance, `-1` = none (the sandbox has no maintenance).
- Example pattern: `0_4_1` is `down` (opened by the operator) and its sibling
  `1_4_4` is at `rho: 1.1737, overflow: true` — keep it hot a third step and
  1_4_4 trips. That's the pattern to catch early. After a trip a line reads
  `status: "down", cooldown: 9` and counts down to 0 (10 steps total).

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
`Observation` attributes the backend reads (verified in 1.12.4):

- `obs.rho` — float array [20], loading per line (same as `lines[].rho`)
- `obs.line_status` — **bool** array, `True` = connected (C3 renders it as
  the `status` string)
- `obs.gen_p` / `obs.load_p` — MW per gen / per load
- Redispatch limits: grid2op has `obs.gen_margin_up` / `gen_margin_down`,
  but **the backend does NOT expose them** — neither `observe` nor
  `--detailed` carries margins or ramp limits. Use the static per-step ramps:
  gen_1_0 5 MW, gen_2_1 10 MW, gen_0_5 15 MW (renewables 0). A single move
  beyond the ramp is rejected as `Ambiguous action` (reward 0). Track your own
  cumulative total per gen, since nothing in the state shows it.
- `obs.current_step` / `obs.max_step` — t / horizon
- `obs.grid_layout` — `{sub_name: (x, y)}`, what `render` draws from
- `obs.get_time_stamp()` — the sim clock as a datetime
- `obs.timestep_overflow` — per line, consecutive steps already spent >1.0
  (not in `--detailed`; verified: 1 after the first hot step, 2 after the
  second; the third hot step trips the line)
- `obs.time_before_cooldown_line` — per line, steps left of reconnection
  cooldown (10 right after a trip; 0 = free) = `lines[].cooldown`
- `obs.time_next_maintenance` — per line, steps to next maintenance (-1 none)
- `obs.simulate(action)` — **free dry-run**: feeds an action through the
  forecast and returns `(next_obs, reward, done, info)` WITHOUT advancing the
  real grid. The backend uses it: `predicted_disc_lines` in `act --json`
  (`verbose`) is its prediction of trips caused by your act.

You don't call these directly (the backend owns the env) — they explain
where every JSON field comes from and what the sim knows.
