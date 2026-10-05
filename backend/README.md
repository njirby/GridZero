# backend (WS A)

FastAPI app that owns the grid2op sim (single writer) and implements the
contracts: C2 (SIM-API), C4 (EVENT-STREAM SSE), and drives opencode (C5).

## Layout
```
backend/app/
  __init__.py
  c3.py              grid2op -> GRID-STATE (C3) mapping (canonical copy)
  sim_session.py     SimSession — the single-writer owner of the env (thread-safe)
  event_bus.py       seq counter, ring buffer, subscribers, STATE/LOG, JSONL log
  opencode_driver.py drives `opencode serve` (C5), translates raw SSE -> C4 agent.*
  main.py            FastAPI app: all routes, control plane, SSE endpoint
backend/tests/
  conftest.py        TestClient fixture (OPENCODE_DISABLE=1 so tests don't spawn opencode)
  test_c3.py         C3 mapping + schema validation + reward semantics
  test_api.py        every C2 route (envelope, illegal, render, meta, attack-501)
  test_sse.py        EventBus (seq, STATE/LOG split, resume) + /event route exists
  test_control.py    pause/resume/take_over/release/single_step/instruction
```

## Run
```
cd /home/nate/grid2op-harness
OPENCODE_DISABLE=1 ./.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8731
# with the agent (opencode must be on PATH):
./.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8731
```
- `OPENCODE_DISABLE=1` runs the sim + SSE + controls WITHOUT the model (for tests / sim-only).
- Without it, the backend starts `opencode serve` on :4097, creates a session, and
  kicks off the agent with the `AGENTS.md`-referencing prompt.

## Endpoints
- C2 (SIM-API): `GET /sim/status`, `GET /sim/state`, `POST /sim/reset`,
  `POST /sim/step`, `POST /sim/act`, `POST /sim/render`, `POST /sim/attack`(501).
- C4 (SSE): `GET /event` (and `/api/event`).
- REST: `GET /state`, `GET /api/grid/meta`.
- Control: `POST /control` {cmd: pause|resume|instruction|single_step|take_over|release|reset|manual_action}.

Envelope on every C2 response: `{"ok","data","error","verbose"}`; HTTP 200 even
when the sim rejects an action (the model reads the body), 400 malformed, 501 attack.

## Test
```
cd /home/nate/grid2op-harness && ./.venv/bin/python -m pytest backend/tests -q
```
25 tests, ~20s (first `grid2op.make` is the slow part; the sim is reused).

## Notes / caveats
- SSE-over-HTTP is validated by booting uvicorn (TestClient hangs on the infinite
  stream — a known starlette+TestClient portal artifact). `test_sse.py` covers the
  EventBus that drives the stream instead.
- Reward: `reward` = most recent step reward (reset value at t=0); `cum_reward` =
  sum of completed steps only (0 at t=0). See C3.
- All env access is under `SimSession._lock` (single writer). Calls from async
  routes go through `asyncio.to_thread`.
