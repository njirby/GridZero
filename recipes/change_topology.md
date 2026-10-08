# Recipe: re-route with bus changes

**When:** a line is hot but you don't want to open it (or can't — it's the
only path). Each substation has **2 busbars**: busbar 1 carries the main
flow, busbar 2 the backup. Moving a line END (or a gen/load) to the other
busbar splits the sub's flow onto its sibling lines — the line stays closed
but carries (nearly) nothing.

## `change_bus` — flip an element to the other busbar

Values are a **LIST of names** (not a name→true dict — that is rejected as
`Illegal — … AmbiguousAction`, and time does not advance):

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
is enough. Only `1` or `2` are legal (bus `3` → rejected, nothing happens,
no step).

**One substation per act.** Two bus changes at different subs in one act
(e.g. `change_bus` on 1_4_4 and 5_12_9) print `Illegal action: … More than 1
substation affected` — the step advances and pays 0.

## Worked re-route (real run, chronic 0)

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
of the top loads and the overload is gone. The MW you moved went somewhere:
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
  it doesn't cut the line. The line reads `status: "up"`, `rho ≈ 0`.
- Re-routes are second-order: every bus move heats some sibling line. Re-
  observe and watch `new_overloads` before the next move.
- **Don't stack bus changes on the same substation.** If a re-route pushes a
  sibling over 1.0, UNDO the maneuver (flip back, re-close the line you
  opened) rather than adding a third topology change. One structural idea
  per turn.
