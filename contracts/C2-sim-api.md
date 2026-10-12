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
not the status) **except** `400` for a non-object `action`, `403`
(`forbidden: token required`) on a token-gated route without the token, and
`5xx` on backend errors (simctl exits 2 on 5xx).

## Endpoints

### `GET /sim/status`
```json
{"ok":true,"data":{"up":true,"env":"l2rpn_case14_sandbox","t":0,"max_t":24,
 "reward":-10.0,"cum_reward":0.0,"done":false},"error":null,"verbose":{}}
```

The grid comes from the backend's `SIM_ENV` env var (default
`l2rpn_case14_sandbox`; e.g. `SIM_ENV=l2rpn_idf_2023`), reported as `env` in
`/sim/status` and C3. `SIM_CHRONIC` / `SIM_HORIZON` / `SIM_SEED` pin the
startup episode.

**Hidden horizon** (`SIM_HIDE_HORIZON=1`, default off): the agent-facing
routes `/sim/status`, `/sim/state` and `/sim/changes` omit `max_t` (it is
optional in the C3 schema), and the `restart_agent` resume prompt drops it.
Step/act outcomes never carried it. `done` and `cause` still report the end
(`Episode is over (time_exceeded)` after the last step). Operator routes
(`/state`, `/event`, `/sim/reset`, `/bench/*`) keep it.

### `GET /sim/state`
Returns the full **GRID-STATE (C3)** object in `data`.
```json
{"ok":true,"data":{ ...C3... }}
```
Query `view=full` (sent by `simctl observe --detailed` and `observe --json`,
i.e. whenever the whole C3 is shown to the model) additionally records this
C3 as the caller's baseline for `/sim/changes`. Without it the call has no
side effect (the compact `simctl observe` and the UI don't move the baseline).

### `GET /sim/changes`
`simctl observe --changes`. Query: `rho_eps` (default `0.01`), `mw_eps`
(default `1.0`); negative → `400`. The backend env vars `SIM_CHANGES_RHO_EPS`
/ `SIM_CHANGES_MW_EPS` replace those defaults (a query param still wins). Returns a **GRID-CHANGES** object
(`contracts/C3-grid-changes.schema.json`): the C3 header fields, all current,
with `lines`/`subs`/`gens` holding ONLY the entities that changed since the
last `view=full` or `/sim/changes` view served to the same viewer, plus:

| field | meaning |
|---|---|
| `full` | `true` = this viewer had no baseline (first call, or first since a reset): every entity is listed |
| `changes_since_t` | `t` of the view this diff is against (`null` when `full`) |
| `lines_changed` / `subs_changed` / `gens_changed` | how many listed (out of `n_line` / `n_sub` / `n_gen`) |
| `thresholds` | `{"rho_eps":…, "mw_eps":…}` actually used |

The baseline lives in the backend, per viewer (keyed by the caller's bearer
token, so a defender and an attacker on the same sim don't share one; `""`
when no token), and is cleared by a reset. `simctl` stays stateless.

**Change rules** (mechanical). An entity is listed IN FULL (every C3 field)
when any of its fields changed vs the version of that entity the viewer was
last shown:

| entity | field | reported when |
|---|---|---|
| line | `status`, `overflow`, `or`, `ex`, `or_bus`, `ex_bus`, `timestep_overflow` | any difference (always, regardless of thresholds) |
| line | `cooldown` | it starts or ends (`0` ↔ `>0`) or restarts (value went up); plain countdown ticks are not changes |
| line | `maint` | its state changes: none (`-1`) / scheduled (`>0`) / now (`0`); countdown ticks are not changes |
| line | `rho` | \|Δ\| ≥ `rho_eps` (0.01 = 1 percentage point) |
| line | `p_or`, `p_ex` | never on their own (rho is the normalized measure of the same flow); carried along |
| gen | `p`, `target_dispatch`, `actual_dispatch` | \|Δ\| ≥ `mw_eps` MW |
| gen | `connected` | any difference |
| sub | `p` (net injection) | \|Δ\| ≥ `mw_eps` MW |
| gen / sub | static fields (`name`, `sub`, `renewable`, `redispatchable`, `type`, `x`, `y`) | any difference |

Only the entities that were listed move the viewer's baseline, so slow drift
is measured against what the viewer last saw and is reported once it
accumulates past the threshold — the viewer's picture of every value is never
off by more than its threshold. C3 has no attack flag: a line cut by an
opponent or a protection trip shows up as its `status` change.
```json
{"ok":true,"data":{"schema_version":1,"env":"l2rpn_case14_sandbox","t":2,"max_t":8064,
 "reward":64.0,"cum_reward":127.4,"done":false,"cause":null,"max_rho":1.012,"n_down":1,
 "n_overflow":1,"overflow_allowed":2,"n_line":20,"n_sub":14,"n_gen":6,"...":"other C3 header fields",
 "full":false,"changes_since_t":1,"thresholds":{"rho_eps":0.01,"mw_eps":1.0},
 "lines_changed":5,"subs_changed":2,"gens_changed":2,
 "lines":[{"id":0,"name":"0_1_0","or":"sub_0","ex":"sub_1","rho":0.5636,"p_or":70.985,
           "p_ex":-69.979,"status":"up","overflow":false,"cooldown":0,"maint":-1,
           "or_bus":1,"ex_bus":1,"timestep_overflow":0}, "..."],
 "subs":["..."],"gens":["..."]},"error":null,"verbose":{}}
```

**Auth (anti reward-hacking):** when the backend is started with
`SIM_API_TOKEN`, the privileged routes require `Authorization: Bearer <tok>`:
`POST /sim/reset` (operator token), `POST /sim/attack` (operator OR attacker
token), `POST /control`, `POST /bench/start`, `GET /bench/stats`,
`POST /bench/attacker` and `/bench/attacker/steer`, and the full-state/stream
routes `GET /event` (`/api/event`) and `GET /state`. `POST /sim/step` with
`n>1` without the operator token is clamped to 1, and a `source` field in
`POST /sim/act` is honored only with the operator token (otherwise the act is
attributed to the agent). Open routes: `/sim/status`, `/sim/state` (the C3 the
model sees), `/sim/changes`, `/sim/act`, `/sim/step` (n=1), `/sim/render`. Without a
configured token (interactive dev mode) everything is open. `reward` at t=0 is
the reset value (-10.0, a placeholder); `cum_reward` excludes it.

### `POST /sim/reset`
Body: `{"env": "l2rpn_case14_sandbox"}` (env optional). Starts a new episode.
Operator-token route (the model's simctl is blocked at the CLI layer too).
```json
{"ok":true,"data":{ ...C3 at t=0... }}
```

### `POST /sim/step`
Body: `{"n": 1}` (n optional, default 1, max 50 with the operator token).
Advances `n` do-nothing steps. Callers without the operator token are clamped
to exactly 1 step. The outcome includes `lines_down`, `overloads` (names of
lines above 1.0 after the step) and `new_overloads`.
```json
{"ok":true,"data":{"t":1,"reward":63.4,"cum_reward":63.4,"done":false,
 "lines_down":0,"overloads":[],"disc_lines":[],"new_overloads":[],
 "illegal":false,"ambiguous":false,"applied":{}},
 "verbose":{"is_illegal":false,"is_ambiguous":false,"opponent_attack_line":null}}
```
After the episode is over (`done`), step returns `ok:false`,
`error:"Episode is over (<cause>). Stop acting and write your summary."` with
the final outcome in `data` (`cause` is `time_exceeded` or `game_over`, also in
C3). `POST /sim/act` behaves the same.

### `POST /sim/act`
Body: `{"action": {<grid2op action dict>}}`. Applies the action AND advances
one timestep. This is the primary action endpoint (model + operator both use it).
Working `action` keys (grid2op 1.12): `set_line_status`, `change_line_status`,
`set_bus`, `change_bus`, `redispatch`, `curtail`. Other keys are silently
ignored. (See `docs/grid2op/action_space.md`.)
```json
{"ok":true,"data":{"t":1,"reward":63.41,"cum_reward":63.41,"done":false,
 "lines_down":1,"overloads":["1_4_4"],"disc_lines":[],"new_overloads":["1_4_4"],
 "illegal":false,"ambiguous":false,
 "applied":{"set_line_status":{"0_4_1":"down"}}},
 "verbose":{"is_illegal":false,"is_ambiguous":false,"opponent_attack_line":null,
 "predicted_disc_lines":[]}}
```
Rejections (`ok:false`, exit 1 in `simctl`):
- grid2op illegal/ambiguous: `error:"Illegal action: <reason>"` (or
  `Ambiguous action: …`); the step DID advance with the action replaced by
  do-nothing (reward 0); `data.illegal:true`.
- action could not be built (bad shape, unknown element):
  `error:"Illegal — <reason>"`; time does NOT advance, `data` carries the
  unchanged `t`/`reward`.

`verbose.opponent_attack_line` is the list of line NAMES a built-in opponent
is attacking on this step (e.g. `["99_105_62"]` on `l2rpn_idf_2023`), `null`
when no attack is active or the env has no opponent (case14). The model's
`simctl act`/`step` text output does not print it.

`disc_lines` lists lines tripped on the step. Known backend bug: it is derived
by using grid2op's `info["disc_lines"]` (cascade-level values per line) as
line ids, so the NAME can be wrong; rely on `lines_down` / C3 line `status`.

### `POST /sim/render`
Body: `{"width": 800, "out": "t0052"}` (both optional). Writes the PNG into the
backend's per-port render dir. `out` must be a bare filename (`.png` appended
if it has no extension); otherwise `ok:false, error:"invalid render filename:
use a bare name like 't0052' or 't0052.png'"`. With `RENDER_DISABLED=1`:
`ok:false, error:"render is disabled in this environment"`.
```json
{"ok":true,"data":{"path":"/home/nate/Documents/GridZero/render/8731/t0052.png",
 "width":800,"height":500,"t":52}}
```

### `POST /sim/attack`  *(operator or attacker token)*
Body: a grid2op action dict (as `/sim/act`; also `{"action": {...}}`). Applies
it as an `opponent` action and advances one step; the defender is not
notified. Same response/rejection shapes as `/sim/act`. 403 without a valid
token.

## Grid2op action-dict reference (what goes in `/sim/act` `action`)

These map to grid2op 1.12 `ActionSpace` construction. **Forms verified against
the installed 1.12.4 by WS C** — several differ from older online docs:

| key | value | meaning |
|-----|-------|---------|
| `set_line_status` | `{"<line>": 1\|-1\|0}` | force line up / down / no-op (dict: name → int) |
| `change_line_status` | `["<line>", ...]` | toggle line status (**LIST of names**, not a dict) |
| `set_bus` | `{"lines_or_id":{"<line>":1\|2}, "lines_ex_id":{...}, "loads_id":{...}, "generators_id":{...}}` | move element to bus N (dict: name → bus) |
| `change_bus` | `{"lines_or_id":["<line>",...], "lines_ex_id":[...], "loads_id":[...], "generators_id":[...]}` | toggle bus (**LIST of names** per sub-key) |
| `redispatch` | `{"<gen>": +ΔMW}` | change gen setpoint by ΔMW (CUMULATIVE; keep within the gen's per-step ramp — the backend does not reject larger values) |
| `curtail` | `{"<gen>": 0.3}` | cap gen at 30% (key is **`curtail`**, NOT `curtailment`) |

**Do NOT use** (silently ignored → the act becomes a no-op, and simctl still prints `applied …`): `curtailment`, `detach_load`, `attach_load`, `set_storage_power`
(this sandbox env's `PlayableAction` drops them with only a Python warning).
See `docs/grid2op/pitfalls.md`.

Line/load/gen names are the exact `env.name_line` / `name_load` / `name_gen`
strings (e.g. `0_1_0`, `gen_1_0`, `sub_0`). The backend pre-validates existence
and, when possible, dry-runs via `obs.simulate(action)`; if the dry-run predicts
a trip, `verbose.predicted_disc_lines` is populated. An unknown/illegal form
comes back `ok:false` with the grid2op reason in `error` (exit code 1 in simctl).
