# Observation — reading the grid state

`simctl observe` is the compact summary you read every turn. For more:
`simctl grid` (= `observe --detailed`) prints every line/sub/gen as a table
row, `observe --changes` prints only the rows that changed, `get` prints every
field of one element, `list` filters/sorts rows, `history` shows an element's
past readings. `--json` on any of them gives the full structured data. The
examples are from a real `l2rpn_case14_sandbox` run (chronic 0); names and
numbers differ on other grids.

## Compact `observe` — the lines

```
t=1 reward=63.4 (cum 63.4) done=no
lines_down=1  max_rho=1.02 (1_4_4)  overloads=[1_4_4]
top loads:  1_4_4=101.9%  5_12_9=80.5%  5_10_7=66.0%
gens: 73.3 72.6 36.6 0.0 0.0 73.0
since your last act (t=0 set_line_status 0_4_1=down): no trips; overloads now: [1_4_4] (max_rho=1.02)
```

- Line 1: step, last step's reward, cumulative reward, episode state. At
  `t=0` the reward is a `-10.0` placeholder (no step has run yet).
- Line 2: count of down lines, the hottest line (name at its rho), and all
  lines above 1.0 right now.
- Line 3: the 3 most loaded LINES, percent of thermal limit (it is a line
  list, not loads).
- Line 4: MW of every generator, in gen id order (the order of the GENS
  table in `simctl grid`, which has the names).
- Line 5: **feedback on your last act** — `(t=<step you acted at> <summary>)`,
  then `no trips` or `tripped: [<lines>]` for the most recent step,
  `; overloads now: [...]` if any line is above 1.0, and `max_rho`. It does
  not exist before your first act. It names the last act even if you have
  since stepped with no-ops; a rejected act's summary ends with
  `(rejected, not applied)`:
  ```
  since your last act (t=8 set_line_status 1_4_4=up (rejected, not applied)): no trips (max_rho=0.77)
  ```
  The authoritative trip check is `lines_down` plus the `down` lines in
  `simctl grid` / `simctl list lines --where status=down`.

## `simctl grid` (= `observe --detailed`) — every element, one row each

```
t=1 reward=63.4 (cum 63.4) done=no · max_rho=1.019 n_down=1 n_overflow=1 overflow_allowed=2 · sim_clock=2019-01-06 00:05:00
since your last act (t=0 set_line_status 0_4_1=down): no trips
LINES (20) — name rho p_or(MW) [flags]; flags only when not default: status (down/cooldown/maintenance), cooldown=N, maint=now|in N, or_bus=/ex_bus= (default 1; -1 when down), overflow k/allowed
  0_1_0     0.579     73.0
  0_4_1     0.000      0.0 down
  1_2_2     0.357     31.7
  1_3_3     0.303     47.5
  1_4_4     1.019     43.9 overflow 1/2
  ...
SUBS (14) — name type net_p(MW, gen positive)
  sub_0  gen       73.0
  sub_1  both      51.2
  ...
  sub_4  load      -6.9
  ...
GENS (6) — name sub p(MW) R=redispatchable W=renewable [flags: disconnected, dispatch target= actual= (MW, when non-zero)]
  gen_1_0 sub_1      73.3 R-
  gen_2_1 sub_2      72.6 R-
  gen_5_2 sub_5      36.6 -W
  ...
(every field of one element: simctl get <line|sub|gen>.<name>; all fields as JSON: simctl --json grid)
```
(`...` marks rows elided here; the real output lists every element.)

- **Header**: step, reward, cumulative reward, `done` (with the cause once
  over), `max_rho`, `n_down` (lines disconnected), `n_overflow` (lines above
  1.0), `overflow_allowed` (overloaded steps a line survives: 2 on case14, it
  trips on the next one), `sim_clock` (the grid's wall clock). Then the
  `since your last act` line (trips only), absent before your first act.
- **Each section header states the columns and the defaults once.** A row
  shows only its columns plus flags for values that are NOT the default.
- **LINES** row: `name rho p_or [flags]`.
  - `name` is `<or_sub>_<ex_sub>_<id>`: `1_4_4` runs sub_1 → sub_4, id 4.
  - **`rho` — the number to watch.** Fraction of the line's thermal limit,
    computed from CURRENT (amps), not from `p_or`: `rho = max(|a_or|,|a_ex|)
    / limit_amps`. **>1.0 = overloaded.** A line trips on its 3rd consecutive
    step >1.0. Don't try to recompute rho from `p_or` — the limit is in amps
    and flows have reactive power, the math won't match.
  - `p_or` — active power (MW) entering at the or end; `> 0` = flow or→ex.
  - Flags: `down` (disconnected — opened by you OR tripped; rho and p_or are
    0), `cooldown` / `maintenance` (the status, when not up/down);
    `cooldown=N` (steps before the line's status may change again — 10 after
    a trip on case14); `maint=now` / `maint in N` (scheduled maintenance);
    `or_bus=2` / `ex_bus=2` (that end is on busbar 2; a down line's ends read
    -1 and are not flagged); `overflow k/allowed` (`k` consecutive overloaded
    steps so far: `overflow 2/2` means the next overloaded step trips it).
- **SUBS** row: `name type net_p`. `type`: `gen`, `load` (pure consumption),
  `both` (gen and load on the same sub), `other`. `net_p` is net MW at the
  sub, generation positive (negative = net import).
- **GENS** row: `name sub p R/W [flags]`. `R` = redispatchable (takes
  `redispatch`), `W` = renewable (can't be redispatched; takes `curtail`;
  its `p` moves with the wind/solar chronic), `-` = no. Flags:
  `disconnected`; `dispatch target=T actual=A` once any redispatch is in
  effect — `target` = your cumulative redispatch request, `actual` = what the
  grid delivers (non-target gens move to rebalance):
  ```
  gen_1_0 sub_1      67.9 R- dispatch target=-5 actual=-4.5
  gen_2_1 sub_2      71.8 R- dispatch target=0 actual=1.75
  ```

Example pattern above: `0_4_1` is `down` (opened by the operator) and its
sibling `1_4_4` is at `overflow 1/2` — two more hot steps and it trips. After
a trip the row reads `1_4_4  0.000  0.0 down cooldown=9` and counts down.

## `--changes` — only what moved

`simctl observe --changes` prints the same section headers and row format as
`grid`, but only for the lines/subs/gens that changed since your last `grid`
(or `observe --detailed`) or `--changes` (the first call, with no earlier
view, shows everything). Discrete changes (status, bus, overflow, a cooldown
starting or ending …) always show; continuous ones only above the thresholds
printed in its header (override with `--rho-eps` / `--mw-eps`). Sections
with no changes are left out.

```
changes since t=1 -> t=2 · reward=63.9 (cum 127.3) · done=no · max_rho=0.791 · n_down=1 · n_overflow=0 · changed: lines 17/20 subs 0/14 gens 1/6 · thresholds rho=0.01 mw=1.0 · overflow_allowed=2
since your last act (t=1 change_bus lines_or_id=['1_4_4']): no trips
LINES changed (17/20) — name rho p_or(MW) [flags]; flags only when not default: …
  1_2_2     0.515     46.2
  1_3_3     0.495     77.5
  1_4_4     0.000     -0.0 or_bus=2
  ...
GENS changed (1/6) — name sub p(MW) R=redispatchable W=renewable [flags: …]
  gen_5_2 sub_5      35.6 -W
```
(header text shortened with `…` here.) With `--json` it is the full-object
JSON of the changed elements only.

## `get` — every field of one element

`simctl get line.<name>`, `get sub.<name>` (also `sub.4`), `get gen.<name>`;
the name may be a glob (`line.0_4_*`). `get state` prints the top-level
fields. Values are exact (unrounded). A sub also lists every attached line
(which end, which busbar, status, rho, p_or) and gen:

```
$ simctl get sub.sub_1
sub sub_1
  id: 1
  name: sub_1
  x: -100.0
  y: -270.0
  type: both
  p: 51.7
  attached lines (4):
    0_1_0 ex end, bus=1, status=up, rho=0.5797, p_or=73.056
    1_2_2 or end, bus=1, status=up, rho=0.5148, p_or=46.201
    1_3_3 or end, bus=1, status=up, rho=0.4953, p_or=77.49
    1_4_4 or end, bus=2, status=up, rho=0.0, p_or=-0.0
  attached gens (1):
    gen_1_0 p=73.9 connected=true
```
An unknown name exits 3 and lists the closest names:
```
$ simctl get gen.gen_1_9
error: unknown gen 'gen_1_9'; closest: gen_1_0, gen_7_4, gen_5_3, gen_5_2, gen_2_1 (`simctl list gens` lists all 6)
```

## `list` — filter and sort rows

`simctl list lines|subs|gens [--sort F] [--asc] [--top N] [--where F=V]...`
prints rows in the `grid` format. `--sort` takes any field of that kind
(descending; `--asc` for ascending). `--where` takes `=` `!=` `>` `<` `>=`
`<=` (quote `<`/`>` in the shell); booleans are `true`/`false`, strings
accept globs; repeat `--where` to AND. A field the row doesn't already show
is appended as `field=value`. An unknown field exits 3 and lists the valid
ones.

```
$ simctl list lines --sort rho --top 3
3 of 20 matching lines (of 20) · sorted by rho descending, top 3 — name rho p_or(MW) [flags]; …
  1_4_4   1.021     44.0 overflow 2/2
  5_12_9  0.776     19.7
  5_10_7  0.633     12.8
$ simctl list lines --where 'cooldown>0'
1 of 1 matching lines (of 20) · where cooldown>0 — name rho p_or(MW) [flags]; …
  1_4_4  0.000      0.0 down cooldown=9
```

## `history` — per-step readings

`simctl history <line|sub|gen>.<name> [--last N]` (default 12) prints one
row per recorded step: line `rho p_or status`, gen `p target_dispatch
actual_dispatch`, sub `p`.

```
$ simctl history line.1_4_4 --last 6
line 1_4_4 — last 6 step(s) (history keeps up to 288)
  t     rho    p_or  status
  4       0       0      up
  5       0       0      up
  6  1.0304  44.368      up
  7  1.0211  43.968      up
  8       0       0    down
  9       0       0    down
```
(t=4–5: or end on busbar 2; t=6–7 overloaded; tripped at t=8, its 3rd
overloaded step.)

## `--json` — the full structured state (C3)

`simctl --json grid` (or `observe --json`) prints the complete GRID-STATE.
These field names are what `get` prints and what `list --sort/--where` accept.

| top level | meaning |
|---|---|
| `t`, `reward`, `cum_reward` | step, this step's reward, episode total |
| `done`, `cause` | episode over? why: `time_exceeded` (ran its full length) or `game_over` (blackout); null = running |
| `delta_min`, `sim_clock` | minutes per step (5.0); the grid's wall clock |
| `n_line`/`n_sub`/`n_gen` | grid size (20 / 14 / 6 on case14) |
| `max_rho`, `n_down`, `n_overflow` | hottest line's loading; lines disconnected; lines above 1.0 |
| `overflow_allowed` | overloaded steps a line survives before tripping |
| `lines`, `subs`, `gens` | one object per element, in id order (fields below) |
| `alarms` | raised alarms |
| `last_action` | the previous act: source, summary, args, t (null before your first act) |
| `last_disc_lines` | lines that tripped on the latest step |
| `env`, `schema_version`, `png` | env name, format version, latest render (if any) |

- **line**: `id`, `name`, `or`, `ex` (end substations), `rho`, `p_or`,
  `p_ex` (MW at each end; `p_or > 0`, `p_ex < 0` for flow or→ex, magnitudes
  differ by the losses), `status` (`up`/`down`/`cooldown`/`maintenance`),
  `overflow` (bool, above 1.0), `cooldown`, `maint` (-1 = none), `or_bus`,
  `ex_bus` (1/2; -1 when down), `timestep_overflow` (consecutive overloaded
  steps so far).
- **sub**: `id`, `name`, `x`, `y` (layout coordinates, what `render` uses),
  `type`, `p`.
- **gen**: `id`, `name`, `sub`, `p`, `renewable`, `redispatchable`,
  `connected`, `target_dispatch`, `actual_dispatch`.

## Advanced `obs` attributes (what the fields come from)

If you want to go deeper than the C3 fields, these are the raw grid2op
`Observation` attributes the backend reads (verified in 1.12.4):

- `obs.rho` — float array [n_line], loading per line (same as `lines[].rho`)
- `obs.line_status` — **bool** array, `True` = connected (C3 renders it as
  the `status` string)
- `obs.gen_p` / `obs.load_p` — MW per gen / per load
- Redispatch limits: grid2op has `obs.gen_margin_up` / `gen_margin_down`,
  but **the backend does NOT expose them** — no view carries margins. Use the per-step ramps listed in "Grid
  facts" in AGENTS.md (renewables 0). A single move
  beyond the ramp is rejected as `Ambiguous action` (reward 0). Your cumulative
  redispatch per gen is `target_dispatch` (what you asked for) and
  `actual_dispatch` (what the grid could deliver): the `dispatch` flag in
  `grid`, or `get gen.<name>`.
- `obs.current_step` — t
- `obs.grid_layout` — `{sub_name: (x, y)}`, what `render` draws from
- `obs.get_time_stamp()` — the sim clock as a datetime
- `obs.timestep_overflow` — per line, consecutive steps already spent >1.0
  = `lines[].timestep_overflow` (the `overflow k/allowed` flag; verified: 1
  after the first hot step, 2 after the second; the third hot step trips
  the line)
- `obs.time_before_cooldown_line` — per line, steps left of reconnection
  cooldown (10 right after a trip; 0 = free) = `lines[].cooldown`
- `obs.time_next_maintenance` — per line, steps to next maintenance (-1 none)
- `obs.simulate(action)` — **free dry-run**: feeds an action through the
  forecast and returns `(next_obs, reward, done, info)` WITHOUT advancing the
  real grid. The backend uses it: `predicted_disc_lines` in `act --json`
  (`verbose`) is its prediction of trips caused by your act.

You don't call these directly (the backend owns the env) — they explain
where every field comes from and what the sim knows.
