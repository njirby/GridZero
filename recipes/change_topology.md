# Recipe: re-route with bus changes

**When:** a line is hot but you don't want to open it (or can't — it's the
only path). Each substation has **2 busbars**: busbar 1 carries the main
flow, busbar 2 the backup. Moving a line END (or a gen/load) to the other
busbar splits the sub's flow onto its sibling lines — the line stays closed
but carries (nearly) nothing.

## `change_bus` — flip an element to the other busbar

Values are a **LIST of names** (not a name→true dict — that raises
`AmbiguousAction` in 1.12.5):

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'
```
Sub-keys: `lines_or_id` (line's "or" end), `lines_ex_id` (its "ex" end),
`loads_id`, `generators_id`.

## `set_bus` — force a specific busbar (1 or 2)

Values are **name → bus number** dicts:

```
simctl act '{"set_bus": {"lines_ex_id": {"3_4_6": 1}}}'
simctl act '{"set_bus": {"lines_or_id": {"0_4_1": 2}}}'
```
Use `set_bus` when you know which side you want; `change_bus` when "flip it"
is enough. Only `1` or `2` are legal (bus `3` → rejected, `AmbiguousAction`).

## Worked re-route (verified at t=0)

State: `0_4_1` open, `1_4_4` overloaded at rho 1.28 (one step from tripping),
sub_4 importing through sub_1.

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'
simctl observe
```
Result: `1_4_4`'s or end sits on sub_1's busbar 2, off the main flow →
`1_4_4` drops to **rho 0.0** (still "up", carrying nothing), sub_4 now
imports via `3_4_6` (**0.56**) — and `4_5_17` (sub_4→sub_5) climbs to
**0.91**. The MW you moved went somewhere: 4_5_17 is your new watch item.

4_5_17 at ~0.91 is now your watch item. Redispatch won't cool it (verified:
pulling gen_2_1 or gen_0_5 leaves it at ~0.91 — it carries lower-ring
transit). If it stays under 1.0 for the next few steps, hold and monitor.
If it crosses 1.0, undo the maneuver (see rules below) — do NOT add a third
topology change.

## Rules of thumb

- A bus change on a line end isolates that end from ONE busbar's elements —
  it doesn't cut the line. The line reads `status: "up"`, `rho ≈ 0`.
- Re-routes are second-order: every bus move heats some sibling line. Re-
  observe and watch `new_overloads` before the next move.
- **Don't stack bus changes on the same substation.** In the scenario above,
  the follow-up "split 4_5_17's ex end" move looks tempting — it is not.
  Verified: isolating 4_5_17's sub_5 end orphans the lower ring's main
  import and overloads FOUR lines at once (5_10_7, 8_9_10, 8_13_11,
  3_8_16). If 4_5_17 crosses 1.0 after your re-route, UNDO the maneuver:
  flip 1_4_4's or end back (`change_bus` again), then re-close 0_4_1.
  One structural idea per turn.
- To undo: the same `change_bus` call flips it back.
