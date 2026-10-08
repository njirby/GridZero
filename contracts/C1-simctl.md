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
Rejections (exit 1): `Illegal action: <reason>` — grid2op replaced the action
with do-nothing, the step STILL advanced, reward 0; `Illegal — <reason>` — the
action could not be built (bad shape / unknown element), time does NOT advance.
Unknown action KEYS are silently ignored and still print `applied <key> …`.
`--json` `data` carries the full step-outcome (C4 `sim.step_outcome` shape).

### `simctl observe [--detailed]`
Current grid state as compact text (default) or full GRID-STATE JSON
(`--detailed` / `--json`). This is the model's "read the state" verb.
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
