# docs/ — model-readable reference for the grid2op harness

Read `../AGENTS.md` first — it's your operator prompt (loop, command table,
concepts). This tree is the detail you pull when you need it.

```
AGENTS.md                     operator prompt — read this first
docs/
  quickstart.md               3-minute start: the loop + one worked example
  grid2op/
    environment.md            episodes: reset/step/done/reward, t, scenarios
    action_space.md           COMPLETE action reference: every key, exact JSON
    observation.md            how to read state: C3 fields + advanced obs attrs
    scoring.md                reward semantics, what "good" looks like
    pitfalls.md               grid2op 1.12 gotchas — read after a rejected act
recipes/
  check_state.md              status + observe
  disconnect_line.md          open a line safely, check new_overloads
  change_topology.md          set_bus / change_bus to re-route flow
  redispatch.md               shift generation
  see_the_grid.md             render + Read the PNG
  full_loop.md                a complete observe→decide→act→verify turn
```

Rules of thumb:

- Every code block in here is a command you can run as-is.
- Numbers in examples are from real `l2rpn_case14_sandbox` runs; your run
  will differ slightly, the patterns won't.
- `simctl` is the only write path to the grid. Docs and recipes are read-only
  context.
