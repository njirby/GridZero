# C2 — SIM-API (HTTP) contract

The HTTP interface between `simctl` (C1) and the backend. The **backend owns
the single grid2op env** (single writer — the only thing that may call
`env.reset`/`env.step`). `simctl` is a remote control over this API. The
frontend also reads from it (state snapshot) but its primary live channel is
the SSE event stream (C4).

Base URL: `http://127.0.0.1:8731` (default). Auth: optional `Authorization:
Bearer <token>`; on loopback the token may be empty. Every mutating endpoint
is serialized by the backend (one in-flight step at a time).

## Envelope

Every response is a single JSON object:
```json
{"ok": true, "data": { }, "error": null, "verbose": { }}
```
- `ok` — `true` on success, `false` on rejection/error.
- `data` — the payload (shape per endpoint below).
- `error` — `null` or a human-readable string (illegal/ambiguous reason, etc.).
- `verbose` — raw grid2op `info`-derived fields (see step endpoints).

HTTP status is `200` for both ok and sim-rejected (the model reads the body,
not the status) **except** `502` when the sim is not running / backend error,
and `400` for malformed request bodies.

## Endpoints

### `GET /sim/status`
```json
{"ok":true,"data":{"up":true,"env":"l2rpn_case14_sandbox","t":50,"max_t":8064,
 "reward":63.1,"cum_reward":3120.4,"done":false},"error":null,"verbose":{}}
```

### `GET /sim/state`
Returns the full **GRID-STATE (C3)** object in `data`.
```json
{"ok":true,"data":{ ...C3... }}
```

**Auth (anti reward-hacking):** when the backend is started with
`SIM_API_TOKEN`, the privileged routes require `Authorization: Bearer <tok>`:
`POST /sim/reset` (operator token), `POST /sim/attack` (operator OR attacker
token), `POST /control`, `POST /bench/start`, `POST /bench/attacker*`.
`POST /sim/step` with `n>1` from unauthenticated traffic is clamped to 1.
Read-only routes (status/state/observe/render/event) stay open. Without a
configured token (interactive dev mode) everything is open.

### `POST /sim/reset`
Body: `{"env": "l2rpn_case14_sandbox"}` (env optional). Starts a new episode.
Operator-token route (the model's simctl is blocked at the CLI layer too).
```json
{"ok":true,"data":{ ...C3 at t=0... }}
```

### `POST /sim/step`
Body: `{"n": 1}` (n optional, default 1, max 50). Advances `n` do-nothing
steps. Unauthenticated callers are clamped to exactly 1 step.
```json
{"ok":true,"data":{"t":51,"reward":63.4,"cum_reward":3121.0,"done":false,
 "lines_down":1,"overloads":[],"disc_lines":[]},
 "verbose":{"is_illegal":false,"is_ambiguous":false,"opponent_attack_line":null}}
```

### `POST /sim/act`
Body: `{"action": {<grid2op action dict>}}`. Applies the action AND advances
one timestep. This is the primary action endpoint (model + operator both use it).
Accepted `action` keys (grid2op 1.12): `set_line_status`, `change_line_status`,
`set_bus`, `change_bus`, `redispatch`, `curtailment`, `set_storage_power`,
`detach_load`, `attach_load`. (See `docs/grid2op/action_space.md`.)
```json
{"ok":true,"data":{"t":52,"reward":61.2,"cum_reward":3074.8,"done":false,
 "disc_lines":[],"new_overloads":["2_3_5"],"illegal":false,"ambiguous":false,
 "applied":{"set_line_status":{"0_4_1":"down"}}},
 "verbose":{"is_illegal":false,"is_ambiguous":false}}
```
If the action is illegal/ambiguous: `ok:false`, `error:"Illegal: ..."`, `data`
still carries the resulting `t`/`reward`, `verbose.is_illegal:true`. Exit-code
1 in `simctl`.

### `POST /sim/render`
Body: `{"width": 800, "out": "t0052"}` (both optional). Writes the PNG.
```json
{"ok":true,"data":{"path":"/home/nate/grid2op-harness/render/t0052.png",
 "width":800,"height":500,"t":52}}
```

### `POST /sim/attack`  *(v1+; absent in v0)*
Body: `{"line":"3_6_15","kind":"trip","duration":5,"source":"user"}`. Applies an
adversarial action and advances one step. The defender model is not notified.
```json
{"ok":true,"data":{"t":53,"attacked":"3_6_15","reward":...}}
```

## Grid2op action-dict reference (what goes in `/sim/act` `action`)

These map to grid2op 1.12 `ActionSpace` construction. **Forms verified against
the installed 1.12.5 by WS C** — several differ from older online docs:

| key | value | meaning |
|-----|-------|---------|
| `set_line_status` | `{"<line>": 1\|-1\|0}` | force line up / down / no-op (dict: name → int) |
| `change_line_status` | `["<line>", ...]` | toggle line status (**LIST of names**, not a dict) |
| `set_bus` | `{"lines_or_id":{"<line>":1\|2}, "lines_ex_id":{...}, "loads_id":{...}, "generators_id":{...}}` | move element to bus N (dict: name → bus) |
| `change_bus` | `{"lines_or_id":["<line>",...], "lines_ex_id":[...], "loads_id":[...], "generators_id":[...]}` | toggle bus (**LIST of names** per sub-key) |
| `redispatch` | `{"<gen>": +ΔMW}` | change gen setpoint by ΔMW (CUMULATIVE; within gen margins or it's flagged ambiguous) |
| `curtail` | `{"<gen>": 0.3}` | cap gen at 30% (key is **`curtail`**, NOT `curtailment`) |

**Do NOT use** (silently ignored → the act becomes a no-op, the model wrongly
believes it acted): `curtailment`, `detach_load`, `attach_load`, `set_storage_power`
(this sandbox env's `PlayableAction` drops them with only a Python warning).
See `docs/grid2op/pitfalls.md`.

Line/load/gen names are the exact `env.name_line` / `name_load` / `name_gen`
strings (e.g. `0_1_0`, `gen_1_0`, `sub_0`). The backend pre-validates existence
and, when possible, dry-runs via `obs.simulate(action)`; if the dry-run predicts
a trip, `verbose.predicted_disc_lines` is populated. An unknown/illegal form
comes back `ok:false` with the grid2op reason in `error` (exit code 1 in simctl).
