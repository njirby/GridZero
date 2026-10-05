# grid2op-harness

A web harness where an LLM (**local Qwen3.5-4B** on vLLM, cards 0-1, via
`opencode`) operates a [grid2op](https://github.com/SimulaTech/grid2op) power-grid
simulation through a terminal-style interface, while a browser UI shows what the
model is doing — its reasoning, each command, its result — and a human can watch,
pause, inject instructions, take over, and attack the grid.

**The trick:** `opencode serve` is already a headless agent runtime (HTTP control
plane + SSE event stream + the model's trained tools: `bash`, `Read`, `Grep`,
`Glob`, `Write`). We don't build an agent loop. We build the **delta**: a sim the
model acts on (`simctl`), docs it reads to learn the API, and a web mirror of its
activity. See **[PLAN.md](PLAN.md)** for the full design, and **[contracts/](contracts/)**
for the 5 pinned interfaces (C1–C5) everything is built against.

## Quick start (once workstreams are merged)

```bash
cd /home/nate/grid2op-harness
source .venv/bin/activate

make backend          # FastAPI: owns grid2op + drives opencode  -> :8731
make web              # Vite dev server (proxies to :8731)       -> :5173
# open http://127.0.0.1:5173
# in the UI: Reset (starts a fresh episode + the model)
```

Or `make serve-all` to run both.

Offline development (no real sim / no model):
```bash
make mock             # mock SIM-API + recorded SSE  -> :8731
make test             # all workstream test suites
```

## Layout

| dir | what | built by |
|-----|------|----------|
| `contracts/` | **the spine** — C1–C5 specs, C3 JSON schema, fixtures, mock SIM-API server | lead (done) |
| `backend/`  | FastAPI app: sim worker (single writer), C2 API, C4 SSE, C5 opencode driver | WS A |
| `cli/`      | `simctl` — the model's terminal (stdlib python) | WS B |
| `web/`      | Vite+React+TS SPA: SVG grid map, agent feed, charts, controls | WS D |
| `docs/` `AGENTS.md` `recipes/` | model-readable grid2op docs + operator prompt | WS C |
| `tests/integration/` | E — run ONE real episode, the v0 acceptance gate | lead |
| `scripts/`  | spine tooling (fixture generator, serve-all) | lead |
| `runs/`     | per-episode event logs + eval reports (gitignored) | runtime |

## Commands

- `make backend` / `make web` / `make mock` / `make serve-all`
- `make test` (all), `make test-a`/`test-b`/`test-c`/`test-d`/`test-e` (per WS)
- `make eval` — run one real episode (needs backend + opencode + docs + simctl)
- `make fixtures` — regenerate `contracts/examples` from a live grid2op

## The model's mental model (what it actually does)

The opencode session's working dir is this repo. `simctl` is on `PATH`. It:
1. reads `AGENTS.md` (auto-loaded) + `docs/` + `recipes/` to learn the sim,
2. `simctl observe` to read the grid (compact text; `--detailed` for full JSON),
3. `simctl act '{"set_line_status":{"0_4_1":-1}}'` to take an action,
4. reads the result (new overloads / rejection reason) and decides the next move,
5. `simctl render` then `Read`s the PNG when it wants to *see* the layout,
6. repeats — keeping the grid stable and the reward high.

The human sees all of it live in the browser (reasoning streams in, each
`simctl` call is a card with its result, the map + reward chart update per step),
and can pause / inject an instruction / take over / issue an adversarial action
the model only perceives as a change in the grid.
