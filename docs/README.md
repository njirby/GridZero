# docs/ — model-readable reference for the grid2op harness

Read `../AGENTS.md` first — it's your operator prompt (loop, command table,
concepts). This tree is the detail you pull when you need it.

```
AGENTS.md                     operator prompt — read this first
docs/
  quickstart.md               3-minute start: the loop + one worked example
  grid2op/
    environment.md            episodes: step/done, info keys, scenarios
    action_space.md           COMPLETE action reference: every key, exact JSON
    observation.md            reading state: every view, row formats, --json fields
    scoring.md                reward semantics, what "good" looks like
    pitfalls.md               grid2op 1.12 gotchas — read after a rejected act
recipes/
  check_state.md              status, observe, grid, get, list, --changes
  disconnect_line.md          open a line safely, check new_overloads
  change_topology.md          set_bus / change_bus to re-route flow
  redispatch.md               shift generation
  see_the_grid.md             optional render + Read the PNG
  full_loop.md                a complete observe→decide→act→verify turn
```

Rules of thumb:

- Every code block in here is a command you can run as-is.
- Numbers and element names in examples are from real `l2rpn_case14_sandbox`
  runs (a small 14-sub grid). Your grid may be different and much bigger:
  its sizes, redispatchable gens + ramps, curtailable gens, reward scale,
  protections and opponent are in the **"Grid facts"** section of AGENTS.md.
  The patterns carry over; the names and numbers don't.
- `simctl` is the only write path to the grid. Docs and recipes are read-only
  context.
