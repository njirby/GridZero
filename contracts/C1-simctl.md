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
| 1    | action rejected by sim (illegal / ambiguous) — `--json` carries the reason |
| 2    | sim error / not running / backend unreachable |
| 3    | unknown command or bad arguments |

On non-zero exit, the **human-readable** reason always prints to **stdout**
(the model reads stdout). With `--json`, a single JSON object prints to stdout
and the reason is in `.error`.

## Commands

### `simctl status`
Is the sim up?
```
sim up · env=l2rpn_case14_sandbox · t=50/8064 · reward=63.1 (cum 3120.4) · done=no
```
`--json`: `{"ok":true,"data":{"up":true,"env":"...","t":50,"max_t":8064,"reward":63.1,"cum_reward":3120.4,"done":false}}`

### `simctl reset [--env NAME]`
New episode. `NAME` defaults to the backend's configured env.
```
reset · env=l2rpn_case14_sandbox · t=0 · reward=64.99
```

### `simctl step [N]`
Advance the sim `N` timesteps (default 1) with **no operator action** (do-nothing
per step). Use to watch natural dynamics.
```
stepped 1 · t=51 · reward=63.4 · lines_down=1 · overloads=[]
```

### `simctl act '<json>'`
Apply a grid2op action **and** advance one timestep (grid2op semantics:
`step` = apply + next frame). The JSON is a grid2op action dict (C2 documents
the accepted keys). This is the primary "take an action" verb.
```
$ simctl act '{"set_line_status":{"0_4_1":-1}}'
applied set_line_status 0_4_1=down · t=52 · reward=61.2 · new_overloads=[2_3_5] · illegal=no
```
`--json` `data` carries the full step-outcome (C4 `sim.step_outcome` shape).

### `simctl observe [--detailed]`
Current grid state as compact text (default) or full GRID-STATE JSON
(`--detailed` / `--json`). This is the model's "read the state" verb.
Default (compact) — lead with the things that matter, flag hazards:
```
t=52 reward=61.2 (cum 3074.8) done=no
lines_down=1  max_rho=0.97 (2_3_5)  overloads=[2_3_5]
top loads:  2_3_5=97.0%  0_4_1=88.0%  5_12_9=71.0%
gens: 81.4 79.3 5.3 0.0 | loads: 5.4 12.6 14.4 ...
since your last act (t=51 set_line_status 0_4_1=-1): 2_3_5 +9.0%, no trips
```
The last line is **feedback on the effect of the model's own last action** —
the single most important line for steering.

### `simctl render [--out PATH] [--width 800]`
Write the grid map as a PNG to `PATH` (default `render/t<NNNN>.png` under the
backend's workspace). Prints the absolute path so the model can `Read` it.
```
wrote /home/nate/grid2op-harness/render/t0052.png (800x500)
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

### `simctl attack <spec>`  *(v1+; absent in v0)*
Apply an adversarial action (a "real" trip/attack, distinct from an operator
`act`). Spec is a JSON like `{"line":"3_6_15","kind":"trip","duration":5}`.
The **defender model is NOT told this happened** — it only sees the effect in
its next `observe`. The backend logs it as `user.action`/`system` for the UI.

## Human-readable vs `--json`

Every command has BOTH. Human text is what the model normally reads (terminal
flavor, one line per fact). `--json` is what tooling/tests read. The human text
must be **stable and greppable** (fixed section order, `key=value` tokens) so
the model can rely on it without parsing JSON.
