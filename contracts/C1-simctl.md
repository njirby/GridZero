# C1 — `simctl` CLI contract

`simctl` is the model's **only** interface to the grid. It is a **stateless
command-line client**: every invocation talks to the SIM-API (C2) over HTTP,
applies one request, prints a result, and exits. No persistent process, no
interactive REPL. This keeps it trivially drivable from opencode's `bash` tool
(and trivially gate-able / mirror-able by the backend).

> The *sim state* is persistent — but it lives in the backend (single writer),
> not in the CLI. The CLI is just a remote control.

## Invocation

```
simctl [--json] [--base-url URL] [--token TOKEN] <command> [args]
```

- `--json`     machine-readable output (default is human-readable text).
- `--base-url` SIM-API base URL (default from `$SIM_API_URL` or `http://127.0.0.1:8731`).
- `--token`    bearer token (default from `$SIM_API_TOKEN`; may be empty on loopback).

Environment: `SIM_API_URL`, `SIM_API_TOKEN`. Both override nothing explicit on
the CLI (explicit flags win).

## Exit codes

| code | meaning |
|------|---------|
| 0    | ok (action applied, or read succeeded) |
| 1    | rejected: illegal/malformed action, `render is disabled`, invalid render filename, or `act`/`step` after the episode ended (`Episode is over (<cause>). Stop acting and write your summary.`) — `--json` carries the reason in `.error` |
| 2    | sim not running (`sim down`) / backend unreachable / HTTP 5xx |
| 3    | unknown command or bad arguments |

On non-zero exit, the **human-readable** reason always prints to **stdout**
(the model reads stdout). With `--json`, a single JSON object prints to stdout
and the reason is in `.error`.

## Commands

### `simctl status`
Is the sim up?
```
sim up · env=l2rpn_case14_sandbox · t=0/24 · reward=-10.0 (cum 0.0) · done=no
```
`--json`: `{"ok":true,"data":{"up":true,"env":"...","t":0,"max_t":24,"reward":-10.0,"cum_reward":0.0,"done":false}}`

### `simctl reset [--env NAME]`
New episode. `NAME` defaults to the backend's configured env. **Operator-only:
`POST /sim/reset` needs the operator token whenever the backend has one, and
the model's session has none (403); with `SIMCTL_NO_RESET=1` the CLI refuses
locally (exit 1). The model can never reset.** Known CLI bug: on a 403 the
command crashes with a Python traceback instead of printing the error.
Not documented to the model.

### `simctl step`
Advance the sim **exactly 1** timestep with **no operator action** (a no-op is
an action). `simctl step N` with N != 1 exits 3 (no multi-step fast-forward;
the backend also clamps `n>1` to 1 for any caller without the operator token).
```
stepped 1 · t=1 · reward=63.4 · lines_down=0 · overloads=[]
```
After the episode ended: `Episode is over (time_exceeded). Stop acting and
write your summary.` (exit 1).

### `simctl act '<json>'`
Apply a grid2op action **and** advance one timestep (grid2op semantics:
`step` = apply + next frame). The JSON is a grid2op action dict (C2 documents
the accepted keys). This is the primary "take an action" verb.
```
$ simctl act '{"set_line_status":{"0_4_1":-1}}'
applied set_line_status 0_4_1=down · t=1 · reward=63.41 · new_overloads=[1_4_4] · illegal=no
```
Rejections (exit 1) are factual explanations that end by saying whether time
advanced (full list and examples in C2 `POST /sim/act`):
`Illegal action: <why>. Not applied: the step advanced to t=N with do-nothing
instead (reward 0).` (or `Ambiguous action: …`) — grid2op rejected it during
the step; `Invalid action: <why>. Not applied; time did NOT advance (t=N).` —
the action could not be built (bad shape, unknown element, out-of-range value).
```
$ simctl act '{"redispatch":{"gen_7_2":-60}}'
Ambiguous action: redispatch gen_7_2 -60 MW exceeds its ramp-down limit 3.5 MW/step (current target_dispatch -3.5 MW). Not applied: the step advanced to t=7 with do-nothing instead (reward 0).
$ simctl act '{"set_line_status":{"x_y":1}}'
Invalid action: unknown line 'x_y' (`simctl list lines` lists them). Not applied; time did NOT advance (t=11).
```
Unknown action KEYS are silently ignored and still print `applied <key> …`.
`--json` `data` carries the full step-outcome (C4 `sim.step_outcome` shape).

### `simctl observe [--detailed]`
Current grid state as a short summary (default). `--detailed` prints exactly
`simctl grid` (below); `--json` prints the full GRID-STATE (C3) envelope. Both
`--detailed` and `--json` also become the baseline for `--changes`. This is
the model's "read the state" verb.
Default (compact) — lead with the things that matter, flag hazards:
```
t=2 reward=64.0 (cum 127.4) done=no
lines_down=1  max_rho=1.01 (1_4_4)  overloads=[1_4_4]
top loads:  1_4_4=101.2%  5_12_9=80.0%  5_10_7=65.7%
gens: 73.9 72.7 35.6 0.0 0.0 71.0
since your last act (t=0 set_line_status 0_4_1=down): no trips; overloads now: [1_4_4] (max_rho=1.01)
```
`top loads` are the 3 hottest LINES; `gens` is MW in `env.name_gen` order. The
last line is **feedback on the model's own last action**: `no trips` or
`tripped: [<lines>]` (from C3 `last_disc_lines`), `overloads now: [...]` when
any line is above 1.0, and `max_rho`. It is omitted before the first act.

### `simctl observe --changes [--rho-eps 0.01] [--mw-eps 1.0]`
Only what changed since your last `observe --detailed` / `grid` (or their
`--json`) or `observe --changes` (`GET /sim/changes`; rules and JSON shape in
C2). The first call, or the first after a reset, is a full dump and says so.
Text output: one header line, the `since your last act` line when there is
one, then the changed elements in the **same row format and section headers
as `simctl grid`** (only sections with changes; every field of an element:
`simctl get`). `--json` gives the GRID-CHANGES object with complete elements.
The thresholds only apply with `--changes` (exit 3 otherwise, and for
`--detailed --changes`). idf_2023 calm step: ~4.5 KB (~1.5k tokens).
```
$ simctl observe --changes
changes since t=6 -> t=7/2017 · reward=211.6 (cum 1453.6) · done=no · max_rho=0.957 · n_down=3 · n_overflow=0 · changed: lines 53/186 subs 24/118 gens 27/62 · thresholds rho=0.01 mw=1.0 · overflow_allowed=3
since your last act (t=6 redispatch gen_7_2=-3.5): no trips
LINES changed (53/186) — name rho p_or(MW) [flags]; flags only when not default: status (down/cooldown/maintenance), cooldown=N, maint=now|in N, or_bus=/ex_bus= (default 1; -1 when down), overflow k/allowed
  69_70_3     0.306    -35.4
  23_71_4     0.325     55.8
  68_76_12    0.549    113.7
  …
SUBS changed (24/118) — name type net_p(MW, gen positive)
  sub_3   both      88.1
  sub_7   both     351.3
  …
GENS changed (27/62) — name sub p(MW) R=redispatchable W=renewable [flags: disconnected, dispatch target= actual= (MW, when non-zero)]
  gen_3_0    sub_3      58.3 -W
  gen_7_2    sub_7     356.0 R- dispatch target=-3.5 actual=-3.4
  …
```

With `SIM_HIDE_HORIZON=1` on the backend, `status` prints `t=<n>` and the
`--changes` header `-> t=<n>` (no `/max_t`), and no JSON output carries
`max_t`; `done=yes` and the `cause` still report the end.

## Query interface (idf_2023-sized grids)

The full C3 is ~90 KB of pretty JSON on idf_2023; these commands read the
same `GET /sim/state` and present it compactly (client-side, stateless).
Every C3 value is reachable through `get` (text) and `--json` (tested on
case14 and idf_2023). Kinds: `line`, `sub`, `gen` (C3 has no loads/storage
entities). Unknown kind/name/field → exit 3 with the valid choices / closest
names.

### `simctl grid`
The whole state as tables: a header line, the `since your last act` line,
then LINES / SUBS / GENS with one row per element and the column header (and
the defaults) once per section. Dropped as derivable/static: `id`, line
`or`/`ex` (in the name) and `p_ex`, sub `x`/`y`. Flags appear only when a
value is not its default. Also the `--changes` baseline (`view=full`).
`--json` → the full C3 envelope. idf_2023: ~11.5 KB, 372 lines.
```
$ simctl grid
t=11 reward=0.0 (cum 1455.6) done=no · max_rho=0.775 n_down=4 n_overflow=0 overflow_allowed=3 · sim_clock=2035-01-01 00:55:00
since your last act (t=10 set_line_status 4_10_162=up (rejected, not applied)): no trips
LINES (186) — name rho p_or(MW) [flags]; flags only when not default: status (down/cooldown/maintenance), cooldown=N, maint=now|in N, or_bus=/ex_bus= (default 1; -1 when down), overflow k/allowed
  0_1_0       0.000      0.0 down
  0_2_1       0.500    -50.8
  …
  21_22_93    0.000      0.0 down cooldown=5
  …
SUBS (118) — name type net_p(MW, gen positive)
  sub_0   load     -50.8
  …
GENS (62) — name sub p(MW) R=redispatchable W=renewable [flags: disconnected, dispatch target= actual= (MW, when non-zero)]
  gen_3_0    sub_3      46.7 -W
  gen_7_2    sub_7     355.4 R- dispatch target=-3.5 actual=-3.5
  …
(every field of one element: simctl get <line|sub|gen>.<name>; all fields as JSON: simctl --json grid)
```

### `simctl get <kind>.<name>` · `simctl get state`
Every field of one element (exact C3 values). `<name>` may be the name, the
id, for subs the bare number (`sub.4`), or a glob (`line.4_10_*`, `gen.*`).
A sub also lists everything attached: each line with which end and which
busbar, and each gen. `get state` prints every top-level C3 field.
`--json` → `{"kind", "objects": [...], "attached": {sub: {"lines", "gens"}}}`.
```
$ simctl get sub.sub_7
sub sub_7
  id: 7
  name: sub_7
  x: -969.0
  y: 131.0
  type: both
  p: 350.9
  attached lines (3):
    7_29_100 or end, bus=1, status=up, rho=0.2692, p_or=131.684
    7_8_129 or end, bus=1, status=up, rho=0.4103, p_or=-66.797
    7_4_173 or end, bus=1, status=up, rho=0.3426, p_or=286.013
  attached gens (1):
    gen_7_2 p=355.4 connected=true
$ simctl get gen.gen_7_3
error: unknown gen 'gen_7_3'; closest: gen_79_43, gen_9_3, gen_7_2, gen_99_53, gen_79_44 (`simctl list gens` lists all 62)
```

### `simctl list lines|subs|gens [--sort FIELD [--asc]] [--top N] [--where EXPR]...`
Rows in the `grid` format. `--sort` any C3 field of that kind (descending;
`--asc` for ascending). `--where FIELD=VALUE` on any field; also `!=`, `>`,
`<`, `>=`, `<=`; booleans `true`/`false`; strings accept globs; repeat
`--where` to AND (quote expressions with `<`/`>` in the shell). A sort/where
field the row doesn't already show is appended as `field=value`. `--json` →
`{"kind", "n_match", "n_total", "objects": [...]}` (full objects).
```
$ simctl list lines --sort rho --top 5
5 of 186 matching lines (of 186) · sorted by rho descending, top 5 — name rho p_or(MW) [flags]; …
  113_114_78  0.775    -50.6
  48_50_136   0.677     32.1
  48_49_135   0.676     25.8
  27_28_99    0.664     45.3
  49_56_147   0.659     17.3
$ simctl list gens --where redispatchable=true --sort p --top 3
3 of 20 matching gens (of 62) · where redispatchable=true, sorted by p descending, top 3 — name sub p(MW) R=redispatchable W=renewable [flags: …]
  gen_25_13 sub_25    359.0 R- dispatch target=0 actual=0.087
  gen_64_35 sub_64    359.0 R- dispatch target=0 actual=0.087
  gen_7_2   sub_7     355.4 R- dispatch target=-3.5 actual=-3.5
```

### `simctl history <kind>.<name> [--last N]`
Per-step readings over the last N recorded steps (default 12, max 288) from
the backend's rolling buffer (`GET /sim/history`): line `rho p_or status`,
gen `p target_dispatch actual_dispatch`, sub `p`. Reset clears it.
```
$ simctl history line.113_114_78 --last 4
line 113_114_78 — last 4 step(s) (history keeps up to 288)
   t     rho     p_or  status
   8  0.9238  -60.669      up
   9  0.8903  -58.501      up
  10  0.8436   -55.37      up
  11  0.7749  -50.635      up
```

### `simctl render [--out NAME] [--width 800]`
Write the grid map as a PNG into the backend's per-port render directory
(default name `t<NNNN>.png`). `--out` must be a **bare filename** (`t0052` or
`t0052.png`); paths / `..` are refused (`invalid render filename: …`, exit 1).
Prints the absolute path so the model can `Read` it. When the backend runs
with `RENDER_DISABLED=1` it prints `render is disabled in this environment`
(exit 1).
```
wrote /home/nate/Documents/GridZero/render/18751/t0000.png (800x500)
```
The model then uses opencode's `Read` tool on that path to *see* the grid.
This is the vision path — **on demand, the model's choice**, not auto-injected.

### `simctl docs`
List the model-readable docs tree (C: `docs/`, `AGENTS.md`, `recipes/`).
```
docs/
  grid2op/
    action_space.md
    observation.md
    environment.md
    scoring.md
  recipes/
    disconnect_line.md
    change_topology.md
    redispatch.md
AGENTS.md   (operator prompt — read this first)
```

### `simctl attack <spec>`  *(attacker session only)*
Apply an adversarial action (a "real" trip/attack, distinct from an operator
`act`). Spec is a JSON like `{"line":"3_6_15","kind":"trip","duration":5}`.
**Gated: only available where `SIMCTL_ATTACKER=1` (the attacker's session) —
elsewhere it exits 3, so the defender model cannot attack its own grid.**
The **defender model is NOT told this happened** — it only sees the effect in
its next `observe`. The backend logs it as `user.action`/`system` for the UI.

## Human-readable vs `--json`

Every command has BOTH. Human text is what the model normally reads (terminal
flavor, one line per fact). `--json` is what tooling/tests read. The human text
must be **stable and greppable** (fixed section order, `key=value` tokens) so
the model can rely on it without parsing JSON.
