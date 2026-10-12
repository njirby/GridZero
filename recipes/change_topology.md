# Recipe: re-route with bus changes

**When:** a line is hot but you don't want to open it (or can't — it's the
only path). Each substation has **2 busbars**: busbar 1 carries the main
flow, busbar 2 the backup. Moving a line END (or a gen/load) to the other
busbar splits the sub's flow onto its sibling lines — the line stays closed
but carries (nearly) nothing.

## `change_bus` — flip an element to the other busbar

Values are a **LIST of names** (a name→true dict is rejected as `Invalid action:
… must be a list of line names`, and time does not advance):

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'
```
Sub-keys: `lines_or_id` (line's "or" end), `lines_ex_id` (its "ex" end),
`loads_id`, `generators_id`.

## `set_bus` — force a specific busbar (1 or 2)

Values are **name → bus number** dicts:

```
simctl act '{"set_bus": {"lines_ex_id": {"3_4_6": 1}}}'
simctl act '{"set_bus": {"lines_or_id": {"1_4_4": 2}}}'
```
Use `set_bus` when you know which side you want; `change_bus` when "flip it"
is enough. Only `1` or `2` are valid busbars (bus `3` → `Invalid action`,
nothing happens, no step).

**One substation per act.** Two bus changes at different subs in one act
are illegal — the step advances and pays 0:
```
$ simctl act '{"change_bus": {"lines_or_id": ["1_4_4", "5_12_9"]}}'
Illegal action: at most 1 substation change per action (got 2: sub_1, sub_5). Not applied: the step advanced to t=5 with do-nothing instead (reward 0).
```

## Worked re-route (real `l2rpn_case14_sandbox` run, chronic 0; names and numbers differ on other grids)

State at t=1: `0_4_1` open, `1_4_4` overloaded at 101.9% (hot for one step,
trips at t=3 if nothing changes), sub_4 importing through sub_1.

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'
simctl observe
```
```
t=2 reward=63.9 (cum 127.3) done=no
lines_down=1  max_rho=0.79 (5_12_9)  overloads=[]
top loads:  5_12_9=79.1%  5_10_7=63.7%  0_1_0=58.0%
gens: 73.9 72.7 35.6 0.0 0.0 73.1
since your last act (t=1 change_bus lines_or_id=['1_4_4']): no trips (max_rho=0.79)
```
`1_4_4`'s or end sits on sub_1's busbar 2, off the main flow, so it drops out
of the top loads and the overload is gone. `simctl get sub.sub_1` shows the
split:
```
  attached lines (4):
    0_1_0 ex end, bus=1, status=up, rho=0.5797, p_or=73.056
    1_2_2 or end, bus=1, status=up, rho=0.5148, p_or=46.201
    1_3_3 or end, bus=1, status=up, rho=0.4953, p_or=77.49
    1_4_4 or end, bus=2, status=up, rho=0.0, p_or=-0.0
```
and its `grid` row reads `1_4_4  0.000  -0.0 or_bus=2`. The MW you moved went somewhere:
`0_1_0` is now the line to watch (58%). In other scenarios the sibling that
takes the load can go hot — check `top loads` every time.

Undo with the same call (`change_bus` toggles back); the line is overloaded
again until `0_4_1` is closed:

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'     # new_overloads=[1_4_4]
simctl act '{"set_line_status": {"0_4_1": 1}}'              # back to lines_down=0
```

## Rules of thumb

- A bus change on a line end isolates that end from ONE busbar's elements —
  it doesn't cut the line. The line stays `up` (no `down` flag), `rho ≈ 0`,
  flagged `or_bus=2` / `ex_bus=2`.
- Re-routes are second-order: every bus move heats some sibling line. Re-
  observe and watch `new_overloads` before the next move.
- **Don't stack bus changes on the same substation.** If a re-route pushes a
  sibling over 1.0, UNDO the maneuver (flip back, re-close the line you
  opened) rather than adding a third topology change. One structural idea
  per turn.
