# Grid2op LLM Harness

A web-app harness where an LLM (vLLM-hosted via the Nuclearn gateway, default
**AA-Dense-Blackwell**) operates a [grid2op](https://github.com/SimulaTech/grid2op)
power-grid simulation. The model drives the sim through a terminal-style
interface (`simctl` + docs it can read + a render it can `Read`), and a browser
UI shows what the model is doing — its reasoning, each command, its result —
while a human can watch, pause, inject instructions, take over, and attack.

## The core realization

`opencode serve` is already a headless agent runtime: an HTTP control plane
(create session, prompt, abort, shell) + an SSE event stream (streaming text,
reasoning, tool calls, results, run status), and the model drives it with tools
it is trained on — `bash`, `Read` (takes images), `Grep`, `Glob`, `Write`.

So we do **not** build an agent loop, a tool-call parser, streaming plumbing, or
an approval system. We build the **delta**:

1. a sim the model can act on,
2. docs the model can read to learn the API,
3. a web mirror of what the model is doing + human controls.

```
            +------------------------------------------------------+
            |  BROWSER (React): grid map + agent feed + controls   |
            +-------------------------^----------------------------+
                                      | SSE (C4) + REST
            +-------------------------+----------------------------+
            |  BACKEND (FastAPI, single process)                   |
            |  - owns the grid2op env (single writer)              |
            |  - implements SIM-API (C2)  <--- simctl (C1) -------|
            |  - drives opencode serve (C5)                       |
            |  - merges agent events + sim events -> SSE (C4)     |
            +----------+-------------------+----------------------+
                       |                   |
                +------v------+      +-----v------+
                | opencode     |      | grid2op env |
                | (model+tools)|      | (pandapower)|
                +------+------+      +------------+
                       |
                model types `simctl ...` via bash,
                `Read`s docs + the render png
```

## The 5 contracts (the spine)

Everything hangs off `contracts/`. **Nothing fans out until these are pinned.**

| # | Contract | File | Purpose |
|---|----------|------|---------|
| C1 | `simctl` CLI | `contracts/C1-simctl.md` | the model's only grid interface (stateless CLI) |
| C2 | SIM-API (HTTP) | `contracts/C2-sim-api.md` | simctl <-> backend; backend owns the env (single writer) |
| C3 | GRID-STATE JSON | `contracts/C3-grid-state.schema.json` | one shape shared by `observe` output AND the frontend map |
| C4 | EVENT-STREAM (SSE) | `contracts/C4-event-stream.md` | backend -> browser; merged agent + sim events, `seq`-numbered |
| C5 | OPENCODE-DRIVER | `contracts/C5-opencode-driver.md` | backend -> `opencode serve`; control plane + event translation |

`contracts/examples/` holds **fixtures** (real `GRID-STATE` snapshots, a canned
`EVENT-STREAM.ndjson`, example `simctl` responses) and `contracts/mock/` is a
**mock SIM-API server** that replays them. The frontend, the CLI, and their
tests run 100% offline against these — that is what makes parallel work safe.

## Repo layout

```
grid2op-harness/
  PLAN.md                  # this file
  README.md                # how to run everything
  .venv/                   # python env: grid2op 1.12.5, numba, matplotlib
  contracts/               # THE SPINE (read-only for subagents A-D)
    C1-simctl.md ... C5-opencode-driver.md
    C3-grid-state.schema.json
    examples/              # fixtures
    mock/mock_sim_server.py
  cli/                     # (B) simctl + tests/cli
  backend/                 # (A) FastAPI app: sim session, SSE, opencode driver
    app/  tests/
  web/                     # (D) Vite+React+TS SPA + tests/web
  docs/  AGENTS.md  recipes/  # (C) model-readable grid2op docs + operator prompt
  tests/integration/       # (E) eval harness: run ONE real episode
  runs/                    # per-episode event logs (gitignored)
  scripts/                 # spine tooling (fixture gen, etc.)
```

## Parallel work breakdown

**Phase 0 — spine (sequential, done by the lead):** C1–C5 specs + schemas +
fixtures + mock server. Everyone else reads this read-only.

**Phase 1 — four independent workstreams, run as parallel background subagents.**
Disjoint directories, so no file collisions. Each owns its own test subdir, so
the test suites are written in parallel, contract-driven.

| WS | Owns | Depends only on | Parallel with |
|----|------|-----------------|---------------|
| **A · Backend** | `backend/` + `backend/tests/` | C2, C3, C4, C5 + real grid2op | B, C, D |
| **B · simctl CLI** | `cli/` + `cli/tests/` | C1, C2 (vs mock SIM-API) | A, C, D |
| **C · Docs + prompt** | `docs/`, `AGENTS.md`, `recipes/` | C1 + grid2op API | A, B, D |
| **D · Frontend** | `web/` + `web/tests/` | C3, C4 + fixtures (offline) | A, B, C |

**Phase 2 — integration/eval (after A+B+C land):** E in `tests/integration/`
spins the real backend, points opencode at the workspace (simctl + docs), runs
**one real episode with AA-Dense-Blackwell**, captures transcript + grid-state
timeline + score. This is the acceptance gate for v0. Frontend flips fixtures ->
live SSE here.

**Phase 3 — features:** v1 (live SVG map, charts, pause/resume/inject/step/
take-over, operator `act`/`attack`, permission gating, `seq` reconnect) -> v2
(take-over/release + replay, speed, bubblewrap sandbox, replay scrubber) -> v3
(two-agent defender-vs-attacker, both blind; native grid2op opponent-attack
hooks; episode scoring; cross-episode memory).

## Milestones

- **v0** — model runs the grid end-to-end and explains itself in a browser.
  (A-min + B + C + D-min [transcript feed + render image] + E [one episode].)
  No SVG map yet (the PNG render is the visual), no charts yet.
- **v1** — live SVG map (C3-driven), reward/stress charts, full control plane,
  operator `act` + `attack` (human is adversarial, model only sees the effect),
  permission gating, `seq` reconnect.
- **v2** — take-over/release + condensed replay, speed control, bubblewrap
  sandbox, event-log replay scrubber.
- **v3** — two-agent adversarial (two blind opencode sessions on one env),
  native opponent-attack hooks, episode scoring/leaderboard, cross-episode memory.

## Locked decisions (lead judgment)

1. **simctl-over-HTTP** (stateless CLI, backend owns the env). opencode's `bash`
   may not hold a persistent interactive process across tool calls; a stateless
   CLI + single-writer backend removes that risk and makes gating + mirroring
   trivial. (An in-process Python REPL is the "purest terminal" but riskier.)
2. **Sandbox (bubblewrap) deferred to v2.** For v0 the project dir + venv +
   simctl-as-only-grid-write-path is enough isolation.
3. **Start env: `l2rpn_case14_sandbox`** (14 sub / 20 line, ~58ms/step, data
   already at `~/data_grid2op`). Bigger L2RPN grids later.
4. **Bind `127.0.0.1`**, optional static bearer token, no auth framework (v0/v1).
5. **Frontend: Vite + React 18 + TS + zustand**, hand-rolled SVG map + charts
   (no d3/plotly/chart.js).
6. **Vision is the model's choice, not a bias.** `simctl render` writes a PNG the
   model can `Read`; `simctl observe` returns text. The model picks per situation.
   Confirmed: AA-Dense-Blackwell reads the map correctly (counted all 14 subs,
   read the 97.75% max loading); an image costs ~676 tokens but a vision turn
   takes ~40s, so "look" is naturally a deliberate, occasional act.

## Validated ground truth (from live probing, 2026-10-02)

- grid2op **1.12.5** is a restructure ("glop"): **no `env.grid`, no
  `env.make()`, no `verbose=True`**. `env.step(action)` -> `(obs, reward, done,
  info)`. `obs` is attribute access (`obs.rho`, `obs.line_status` [bool],
  `obs.gen_p`, `obs.current_step`, `obs.grid_layout`, `obs.to_json()`).
- Actions: **both** a chaining factory (`A.change_bus(name, extremity="or",
  previous_action=act)`) **and** a dict form (`A({"set_line_status":
  {"0_1_0": -1}})`) work. `set_line_status` value: `1`=force up, `-1`=force
  down, `0`=noop; `change_bus` needs `extremity` in 1.12.
- `env.attach_renderer()` then `env.render()` -> matplotlib `Figure` -> PNG.
  ~750ms per render (fine on-demand, too slow per-step). `obs.grid_layout` ->
  `{sub_name: (x, y)}` for our own SVG.
- `obs.simulate(action)` is a **free dry-run** (predict trips before committing).
- **Built-in opponent-attack hooks** exist in `info`: `opponent_attack_line`,
  `opponent_attack_sub`, `opponent_attack_duration`, `time_since_last_attack`.
- Gateway: `https://llmgateway.nuclearn.com/v1` (LiteLLM -> vLLM 0.26.0), model
  id `AA-Dense-Blackwell`, key in `~/.local/share/opencode/auth.json`. Tool
  calling works (OpenAI format, args always valid JSON). **It is a reasoning
  model**: `reasoning_content` burns into `max_tokens` — budget >= 2000/turn.
- `opencode serve` (v1.18.34): `POST /api/session`, `POST /session/{id}/prompt`
  (blocking) / `prompt_async`, `POST /session/{id}/abort`, `POST
  /session/{id}/shell`, `GET /event` (SSE: `message.part.delta`, `reasoning`,
  `tool`, `message.part.updated`, `session.status` busy/idle). Root `/` ships a
  web UI.

## Testing strategy (parallel-friendly)

- **Contract tests** from Phase 0: every C has a spec + fixtures; tests assert
  responses/streams match. Written per-WS in parallel.
- **B** (CLI) vs mock SIM-API (offline). **A** (backend) real grid2op smoke
  (reset/step/act/render determinism). **D** (web) component + recorded-stream
  playback (offline). **E** (integration) one real episode: model produced >=1
  legal act, transcript shows reasoning+tool+outcome, grid state advanced.
