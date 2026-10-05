# simctl (WS B)

The model's terminal for the grid2op harness. A **stateless** stdlib-Python CLI:
each invocation makes one HTTP call to the SIM-API (C2) and exits. Driven from
opencode's `bash` tool.

## Usage
```
simctl [--json] [--base-url URL] [--token TOKEN] <command> [args]

simctl status
simctl reset [--env l2rpn_case14_sandbox]
simctl step [N]
simctl act '<grid2op action json>'        # e.g. '{"set_line_status":{"0_4_1":-1}}'
simctl observe [--detailed]
simctl render [--out PATH] [--width 800]
simctl docs [--docs-dir DIR]
simctl attack '<json>'                    # v1+ (stub in v0)
```

## Env vars
- `SIM_API_URL` — SIM-API base (default `http://127.0.0.1:8731`)
- `SIM_API_TOKEN` — bearer token (may be empty on loopback)

## Exit codes
| code | meaning |
|------|---------|
| 0 | ok (action applied / read succeeded) |
| 1 | action rejected by sim (illegal/ambiguous) |
| 2 | backend unreachable / sim error |
| 3 | unknown command / bad arguments |

The human-readable reason always prints to **stdout** (the model reads stdout).
`--json` prints the full C2 envelope to stdout instead.

## Run the tests
```
cd /home/nate/grid2op-harness && python3 -m pytest cli/tests -q
```
Tests start the mock SIM-API (contracts/mock) on port 8733 and exercise every
command against it — no real grid2op needed.
