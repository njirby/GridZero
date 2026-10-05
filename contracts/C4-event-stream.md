# C4 — EVENT-STREAM (SSE) contract

The backend -> browser live channel. One SSE endpoint. The backend **merges**
two sources into a single ordered stream: (1) sim events (state, step outcomes)
and (2) agent events (translated from opencode's raw SSE, see C5). The browser
is a dumb renderer of this envelope — it never talks to opencode directly.

## Endpoint

`GET /event`  (or `/api/event` alias). Standard SSE:
```
id: 482
data: {"seq":482,"type":"sim.state","ts":1759344000.123,"data":{...}}

id: 483
data: {"seq":483,"type":"agent.delta","ts":...,"data":{...}}

```
- Every frame carries an SSE `id:` equal to `seq`, so the browser's
  `EventSource` auto-resends `Last-Event-ID` on reconnect.
- Heartbeat: a `{"type":"ping"}` frame every 15s (no `id`).

## Envelope

```json
{"seq": 482, "type": "<event-type>", "ts": 1759344000.123, "data": { ... }}
```
`seq` is a **global monotonic int** across all types (single counter in the
backend). `ts` is server unix seconds.

## Event classes

- **STATE** — latest-wins, replaceable, droppable under load. The browser keeps
  only the newest per key.
- **LOG** — append-only, replayable, never dropped within the ring.

| type | class | `data` |
|------|-------|--------|
| `sim.state` | STATE | a full **GRID-STATE (C3)** object |
| `sim.step_outcome` | LOG | per-step "what happened" (below) |
| `session.status` | STATE | `{"status":"busy"\|"idle"\|"error","title":"..."}` |
| `agent.delta` | LOG | streaming text/reasoning chunk (below) |
| `agent.tool_call` | LOG | a tool invocation began (below) |
| `agent.tool_result` | LOG | a tool invocation finished (below) |
| `agent.turn_end` | LOG | one model turn finished (below) |
| `user.action` | LOG | echo of every human command (below) |
| `opponent.action` | LOG | a manual/human attack applied via `POST /sim/attack` (below) |
| `opponent.step` | LOG | a scheduled/scripted attack fired by the backend at a sim step (below) |
| `system` | LOG | `{"level":"info"\|"warn"\|"error","msg":"..."}` |
| `episode.summary` | LOG | end-of-episode rollup (below) |
| `ping` | — | `{}` |

## `sim.step_outcome`
The precomputed "why/what happened" per tick. This is the consequence side of
the trace (paired with the model's intent from `agent.*`). `source` is
`agent` (the LLM/human via simctl), `auto` (no-op step), `user` (operator action),
or **`opponent`** (an adversarial attack — the defending agent is blind to who
acted; it only sees the grid effect).
```json
{"t":52,"source":"agent",
 "action":{"tool":"simctl act","args":{"set_line_status":{"0_4_1":-1}},"summary":"0_4_1 down"},
 "reward":61.2,"cum_reward":3074.8,"done":false,
 "disc_lines":["2_3_5"],"illegal":false,"ambiguous":false,
 "new_overloads":["2_3_5"],"predicted_disc_lines":[]}
```

## `opponent.action`
A manual attack fired through `POST /sim/attack` (the web-UI "ATTACK" panel, or
a human/second-agent). The defender is never told who attacked — it only sees the
grid change in its next observe. Emitted by the backend right after the attack.
```json
{"action":{"set_line_status":{"0_4_1":-1}}, "summary":{"set_line_status":{"0_4_1":"down"}}}
```

## `opponent.step`
A scheduled/scripted adversarial attack fired by the backend at an exact sim step
(adversarial/robustness benchmark). Pace-independent (fires regardless of how fast
the defender advances). `kind` is `start` (line cut) or `end` (line restored).
```json
{"t":50,"kind":"start","line":"0_4_1","action":{"set_line_status":{"0_4_1":-1}}}
```

## `agent.delta`
Streaming token. `field` discriminates the channel (one event type, not three).
```json
{"turn":"t-9f2","message_id":"msg_...","part_id":"prt_...","field":"text","delta":"Line "}
```
`field` ∈ `text` | `reasoning`. The browser accumulates deltas per `part_id`
and flushes to the UI once per animation frame.

## `agent.tool_call`
```json
{"turn":"t-9f2","part_id":"prt_...","tool":"bash",
 "input":{"command":"simctl act '{\"set_line_status\":{\"0_4_1\":-1}}'"}}
```

## `agent.tool_result`
```json
{"turn":"t-9f2","part_id":"prt_...","tool":"bash",
 "status":"completed",
 "input":{"command":"simctl act '{\"set_line_status\":{\"0_4_1\":-1}}'"},
 "output":"Line 0_4_1 opened. Overload on 2_3_5 (102%). Δreward -1.20", "duration_ms":84}
```
`status` ∈ `completed` | `error` | `pending`. `input` is carried here too because the
pending `agent.tool_call` often arrives with an empty `{}` and the command only fills in
on the completed part — the UI should prefer `tool_result.input` when `tool_call.input` is empty.

## `agent.turn_end`
```json
{"turn":"t-9f2","tokens_in":1842,"tokens_out":210,"reasoning_tokens":160,
 "latency_ms":2310,"cost_usd":0.0061,"final_action":{"tool":"bash","summary":"simctl act ..."}}
```

## `user.action`
```json
{"id":"uuid","cmd":"manual_action","args":{"tool":"simctl act","args":{"set_line_status":{"5_10_7":-1}}}}
```
Also emitted for `instruction`, `pause`, `resume`, `single_step`, `take_over`,
`release`, `reset`.

## `episode.summary`
```json
{"t":214,"cum_reward":13882.1,"cause":"time_exceeded","peak_max_rho":0.99,
 "n_down":3,"n_topology_changes":11,"n_user_actions":2,"n_instructions":1,
 "duration_s":412.7,
 "llm":{"turns":210,"tokens_in":391200,"tokens_out":48110,"cost_usd":1.13}}
```

## Reconnect / resume

- Browser holds `lastSeq`. On `EventSource` reopen it sends
  `Last-Event-ID: <lastSeq>` (native SSE) — the backend replays LOG events with
  `seq > lastSeq` from its ring buffer (default 5000), then sends the current
  STATE snapshot, then continues live.
- If `lastSeq` is older than the ring (gap), the backend emits
  `{"type":"system","data":{"level":"warn","msg":"gap","from":N,"to":M}}` and
  sends a fresh `sim.state` + `session.status` so the browser resyncs; the
  browser may also `GET /event?after_seq=M` is NOT needed — the ring handles it.
- `GET /state` (REST) returns the current STATE snapshot as JSON (poll fallback).

## REST (non-SSE) endpoints on the backend

- `GET /state`      -> `{"sim": <C3>, "mode":"agent", "running":true, "last_seq":N}`
- `GET /sim/state`  -> C3 (C2)
- `POST /control`   -> `{"cmd":"pause|resume|single_step|take_over|release|reset|instruction","args":{}}` (see C5 for what the backend does with each)
- `GET /api/grid/meta` -> static per-env: sub coords/types, line endpoints, thermal limits, `grid_layout`. Fetched once; positions never change. (Backend also serves the C3 `subs`/`lines` live; meta is for labels/limits.)
