# simctl (WS B)

The model's terminal for the grid2op harness. A **stateless** stdlib-Python CLI:
each invocation makes one HTTP call to the SIM-API (C2) and exits. Driven from
opencode's `bash` tool.

## Usage
```
simctl [--json] [--base-url URL] [--token TOKEN] <command> [args]

simctl status
simctl step                               # exactly 1 step; `step N` (N>1) exits 3
simctl reset [--env ...]                  # OPERATOR ONLY (see below) — not for the model
simctl act '<grid2op action json>'        # e.g. '{"set_line_status":{"0_4_1":-1}}'
simctl observe [--detailed]
simctl render [--out NAME.png] [--width 800]   # --out = bare filename; may be disabled
simctl docs [--docs-dir DIR]
simctl attack '<json>'                    # attacker session only (SIMCTL_ATTACKER=1)
```

## Env vars
- `SIM_API_URL` — SIM-API base (default `http://127.0.0.1:8731`)
- `SIM_API_TOKEN` — bearer token (may be empty on loopback). The model's
  session does NOT get the operator token; with a token configured on the
  backend, `/sim/reset`, multi-step `/sim/step`, `/event`, `/state` and
  `/bench/*` need it, so `simctl reset` from the model gets 403. (Known bug:
  `simctl reset` then crashes with a traceback instead of printing the 403.)
- `SIMCTL_NO_RESET=1` — `simctl reset` refuses locally.
- `SIMCTL_ATTACKER=1` — enables `simctl attack` (attacker session only).

## Exit codes
| code | meaning |
|------|---------|
| 0 | ok (action applied / read succeeded) |
| 1 | rejected: `Illegal action: <reason>` / `Illegal — <reason>`, `render is disabled`, `Episode is over (<cause>). Stop acting and write your summary.` |
| 2 | backend unreachable / 5xx / `sim down` |
| 3 | unknown command / bad arguments |

The human-readable reason always prints to **stdout** (the model reads stdout).
`--json` prints the full C2 envelope to stdout instead.

## Run the tests
```
cd /home/nate/Documents/GridZero && python3 -m pytest cli/tests -q
```
Tests start the mock SIM-API (contracts/mock) on port 8733 and exercise every
command against it — no real grid2op needed.
