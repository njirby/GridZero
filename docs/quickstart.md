# Quickstart — 3 minutes to your first act

`simctl` is a stateless CLI on PATH. It talks to the running grid2op sim
(`l2rpn_case14_sandbox`: 14 subs, 20 lines, 6 gens). You operate the grid by
repeating one loop:

```
observe  →  decide (1-2 sentences of reasoning)  →  act  →  read the result  →  observe
```

`act` applies your move AND advances the clock one 5-minute step. `observe`
always shows you the grid AFTER the last step.

## The two commands that matter

```
simctl observe
```
```
t=52 reward=61.2 (cum 3074.8) done=no
lines_down=1  max_rho=0.97 (2_3_5)  overloads=[2_3_5]
top loads:  2_3_5=97.0%  0_4_1=88.0%  5_12_9=71.0%
gens: 81.4 79.3 5.3 0.0 | loads: 5.4 12.6 14.4 ...
since your last act (t=51 set_line_status 0_4_1=-1): 2_3_5 +9.0%, no trips
```

- `max_rho` — the most loaded line. 1.0 = at its thermal limit. **>1.0 for
  two consecutive steps and it trips.**
- `overloads` — lines currently above 1.0.
- Last line — what your previous action did. This is your steering signal.

```
simctl act '{"set_line_status": {"0_4_1": -1}}'
```
```
applied set_line_status 0_4_1=down · t=52 · reward=61.2 · new_overloads=[1_4_4] · illegal=no
```

- `new_overloads` — lines that JUST crossed 1.0 because of your move.
- `illegal=yes` — the move was rejected, the grid stepped as if you did
  nothing, and the reason is printed. Fix the move, try again.

## Worked example: open a line, eat the overload, re-route it

Start from a fresh episode (values from a real run; yours will be close).

**Turn 1 — open line 0_4_1 (sub_0 → sub_4).** sub_4 loses one of its four
feeders (it also has 1_4_4 from sub_1, 3_4_6 from sub_3, 4_5_17 to sub_5);
part of the load has to come through sub_1:

```
simctl act '{"set_line_status": {"0_4_1": -1}}'
simctl observe
```
You'll see `1_4_4` in `overloads` at ~128% (in a real t=0 run: rho 1.277).
That's the load you shifted. Left alone, 1_4_4 trips in one more step and is
out for ~10 steps. Don't leave it there.

**Turn 2 — re-route.** Move 1_4_4's sub_1 end onto sub_1's backup busbar so
sub_4 imports through 3_4_6 (sub_3 side) instead:

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'
simctl observe
```
You'll see `1_4_4` drop back (its or end is now on busbar 2, off the main
flow), `3_4_6` climb to ~56%, and **`4_5_17` climb to ~91%** — the re-route
pushed load onto 4_5_17 (sub_4 → sub_5). 91% is hot; if it stays above 1.0
for two steps it trips too.

**Turn 3 — settle.** 4_5_17 at ~0.91 is hot but under 1.0. Leave it for one
or two steps and watch the `observe`; if it crosses 1.0, don't stack a third
topology move on sub_4 (verified to cascade) — undo the maneuver: flip
1_4_4's or end back, then re-close 0_4_1:

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'   # flip back (same call toggles)
simctl act '{"set_line_status": {"0_4_1": 1}}'
```
A line you opened yourself has no cooldown, so the undo is instant — in the
clean two-step variant (open then close, no bus move) 1_4_4 relaxes back to
~0.83 in one step.

**The habit**: act → observe → check `new_overloads` and `max_rho` → fix the
thing you just heated up before you do anything else. One change per turn,
watch the second-order effect.

## If the sim isn't up

```
simctl status          # "sim down ..." or backend error
simctl reset           # fresh episode, t=0
```

Next: `grid2op/action_space.md` (every action key), `grid2op/pitfalls.md`
(what breaks silently).
