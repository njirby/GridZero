# C5 — OPENCODE-DRIVER contract

How the backend drives `opencode serve` and translates its raw SSE into C4
agent events. `opencode serve` is the agent runtime: it hosts the model
(qwen3.5-4b by default), gives it the tools (`bash`, `Read`, `Grep`,
`Glob`, `Write`, ...), and exposes an HTTP control plane + event stream. The
backend is just a client of that.

> Validated live against opencode v1.18.34 on 2026-10-02. Route spellings
> below are exact — do not guess.

## Lifecycle

1. **Start the server** (backend boots it, or attaches to a running one):
   `opencode serve --hostname 127.0.0.1 --port <P>`  (cwd = the harness
   workspace so the model sees `docs/`, `AGENTS.md`, `simctl` on PATH).
2. **Create a session**:
   `POST /api/session`  body `{"agent":"build","model":{"id":"qwen3.5-4b","providerID":"vllm4b"}}`
   -> `{"data":{"id":"ses_..."}}`.
   **Gotcha (validated):** the model ref key is `id`, NOT `modelID` — with `modelID`
   the server 400s with `Missing key at ["model"]["id"]`. Omitting `model` also works
   (falls back to the config default, which is qwen3.5-4b).
3. **Prime it** (the operator prompt): `POST /api/session/{id}/prompt` (blocking)
   or `prompt_async` with the AGENTS.md-derived kickoff: "You are a grid
   operator. Read AGENTS.md, then run `simctl status` and begin."
4. **Steer** mid-run: `POST /api/session/{id}/prompt_async` with
   `{"parts":[{"type":"text","text":"[OPERATOR] <instruction>"}]}`.
5. **Pause**: `POST /api/session/{id}/abort`.
6. **Watch**: `GET /event` (root, NOT `/api/event`) — SSE of everything.

## Exact opencode routes (validated)

> **Two route families — this bites you.** Session *creation/listing/message* are
> under `/api/session/...`, but firing a prompt and abort/shell are under
> `/session/...` (**no `/api` prefix**). A `parts` body to `/api/session/{id}/prompt`
> 400s (`Missing key at ["prompt"]`); the `/api/.../prompt_async` route 404s (returns
> the SPA HTML). Use the table below verbatim.

| opencode endpoint | method | backend uses it for |
|-------------------|--------|---------------------|
| `/api/session` | POST | create session |
| `/api/session` | GET | list sessions (session list is global) |
| `/session/{id}/prompt_async` | POST | **fire a turn (kickoff + steer/inject)**; body `{"parts":[{"type":"text","text":...}]}` → 204 |
| `/session/{id}/abort` | POST | pause / stop the in-flight turn |
| `/session/{id}/shell` | POST | run a shell command as the model |
| `/api/session/{id}/message` | GET | message history (replay) |
| `/api/session/{id}/history` | GET | history |
| `/api/session/{id}/permission/{reqID}/reply` | POST | approve/deny a permission prompt |
| `/api/session/{id}/todo` | GET | model's todo list |
| `/event` | GET (SSE) | the raw agent event stream |

## Raw opencode SSE -> C4 translation

The backend consumes `GET /event` and maps each frame to a C4 `agent.*` event
(wrapping in the C4 envelope with the global `seq`). Raw opencode frame shape:
`{"id":"evt_...","type":"<T>","properties":{...}}`.

| opencode `type` | properties | -> C4 |
|-----------------|-----------|-------|
| `message.part.delta` | `messageID, partID, field("text"), delta` | `agent.delta` {field: "text", delta} |
| `message.part.updated` where `part.type=="reasoning"` | `part.text`, `part.time` | `agent.delta` {field:"reasoning", ...} / reasoning block |
| `message.part.updated` where `part.type=="tool"` | `part.tool`, `part.state{status,input,output}` | `agent.tool_call` (status pending) then `agent.tool_result` (status completed/error, with `output`, `duration_ms`) |
| `message.updated` | `info{role, ...}` | (bookkeeping; start of a turn -> new `turn` id) |
| `reasoning` | `messageID, partID` | `agent.delta` {field:"reasoning"} |
| `session.status` | `status{type:"busy"\|"idle"}` | `session.status` {status} |
| `session.idle` / `idle` | — | `session.status` {status:"idle"} |
| `step-start` / `step-finish` | — | (turn boundary bookkeeping) |
| `message.part.updated` where `part.type=="text"` (final) | `part.text` | finalize the text part |
| `session.diff` | — | (file changes; ignore for grid, keep for replay) |

`turn` id: derive one per assistant message (`messageID`), propagate through
that turn's deltas/tool events, close with `agent.turn_end` when `session.status`
goes idle (or `step-finish`).

### Tool call / result detail
opencode emits a `tool` part that transitions `state.status`
`pending -> completed|error`. The backend:
- on first sight (`pending`, has `input`) -> C4 `agent.tool_call` {tool, input}
- on `completed`/`error` -> C4 `agent.tool_result` {tool, status, output, duration_ms}
`input.command` for the `bash` tool is the literal `simctl ...` string — this is
what the UI renders as the model's action card, and what the backend greps to
detect "the model acted on the grid" (to pair with the `sim.step_outcome`).

## Control commands (browser -> backend `POST /control`) -> backend behavior

| `cmd` | backend does |
|-------|-------------|
| `pause` | `POST /session/{id}/abort`; set `session.status` idle; emit `user.action` |
| `resume` | `prompt_async` "[OPERATOR] resume; continue from t=N"; emit `user.action` |
| `single_step` | if paused: run one no-op `simctl step` (or the queued action); emit outcome |
| `instruction` | `prompt_async` `{"parts":[{"type":"text","text":"[OPERATOR] <text>"}]}` (poke optional) |
| `take_over` | `abort`; backend goes MANUAL: sim advances only via operator `manual_action`/`step`; model loop stopped |
| `release` | send model a **condensed replay**: last 20 `sim.step_outcome` one-liners + current `sim.state` + "operator was in control t=a..b, they issued: …; you resume now" |
| `reset` | hard: `abort`, new opencode session, `simctl reset`, fresh conversation |
| `manual_action` | apply the given grid2op action via `/sim/act` (source="user"); the model is NOT told who, only sees the effect next `observe` |

Free-text `instruction` is delivered **at the loop boundary** (opencode handles
in-flight queuing; we just `prompt_async` and it lands on the next model step).
Every `cmd` is echoed as a `user.action` C4 event so the feed + replay include
human actions.

## Agent-loop notes

- One opencode turn = one model reasoning step that may emit several tool calls
  (e.g. `simctl observe` then `simctl act`). The backend does NOT pace this —
  the model decides how many simctl calls per turn. Pacing/`wait` is the
  model's choice via `simctl step N`.
- The model's system context = opencode's agent config + the workspace
  `AGENTS.md` (C produces it). The backend does not build a custom system
  prompt; it relies on AGENTS.md + docs/ in the working directory.
- Gating (v1): opencode permission prompts (`/api/session/{id}/permission`)
  let the backend auto-approve read-only `simctl` and optionally gate
  `simctl act`/`reset`/`attack` behind a UI confirm. For v0, `--auto`-equivalent
  (approve all) is fine since the whole action space is in-process sim.
