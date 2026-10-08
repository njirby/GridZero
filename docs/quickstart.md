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
t=2 reward=64.0 (cum 127.4) done=no
lines_down=1  max_rho=1.01 (1_4_4)  overloads=[1_4_4]
top loads:  1_4_4=101.2%  5_12_9=80.0%  5_10_7=65.7%
gens: 73.9 72.7 35.6 0.0 0.0 71.0
since your last act (t=0 set_line_status 0_4_1=down): no trips; overloads now: [1_4_4] (max_rho=1.01)
```

- `max_rho` — the most loaded line. 1.0 = at its thermal limit. **A line above
  1.0 trips on its 3rd consecutive overloaded step** (you get 2 acts to fix it).
- `overloads` — lines currently above 1.0. `top loads` — the 3 hottest lines.
- `gens` — MW of gen_1_0, gen_2_1, gen_5_2, gen_5_3, gen_7_4, gen_0_5.
- Last line — what your previous action did: `no trips` or `tripped: [...]`,
  plus the overloads that exist now. Absent before your first act (and at
  t=0 `reward=-10.0` is a placeholder, not a real step).

```
simctl act '{"set_line_status": {"0_4_1": -1}}'
```
```
applied set_line_status 0_4_1=down · t=1 · reward=63.41 · new_overloads=[1_4_4] · illegal=no
```

- `new_overloads` — lines that JUST crossed 1.0 because of your move.
- A rejected move prints `Illegal action: <reason>` (exit 1) instead. It was
  NOT applied, but the clock still advanced one step and that step paid 0.

## Worked example: open a line, eat the overload, re-route it

Start from a fresh episode (real run, chronic 0; other scenarios differ in
the numbers, not the pattern).

**Turn 1 — open line 0_4_1 (sub_0 → sub_4).** sub_4 loses one of its
feeders (it also has 1_4_4 from sub_1, 3_4_6 from sub_3, 4_5_17 to sub_5);
part of the load has to come through sub_1:

```
simctl act '{"set_line_status": {"0_4_1": -1}}'
simctl observe
```
You'll see `1_4_4` in `overloads` at ~102% (rho 1.019 at t=1). That's the
load you shifted. Left alone it trips on the third overloaded step (t=3) and
is out for 10 steps. Don't leave it there.

**Turn 2 — re-route.** Move 1_4_4's sub_1 end onto sub_1's backup busbar:

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'
simctl observe
```
`overloads=[]` and `max_rho=0.79`: 1_4_4 is now off the main flow (its or end
is on busbar 2) and sub_4 imports through its other feeders. Always check
which sibling took the load; here nothing got hot, in other scenarios it can.

**Turn 3 — undo.** A line you opened yourself has no cooldown, so undoing is
instant. Flip the bus back and re-close 0_4_1:

```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'   # same call toggles back
simctl act '{"set_line_status": {"0_4_1": 1}}'
```
After the flip-back 1_4_4 is overloaded again (`new_overloads=[1_4_4]`)
until 0_4_1 is back; one step later `lines_down=0`, `max_rho=0.80`.

**The habit**: act -> observe -> check `new_overloads` and `max_rho` -> fix the
thing you just heated up before you do anything else. One change per turn,
watch the second-order effect.

## If the sim isn't up

```
simctl status          # "sim down ..." or backend error (exit 2)
```
You CANNOT reset the episode — if the sim is down, retry `simctl status` a
few times (the backend may be restarting); if it stays down, the run is over.
After `done=yes` (`t` reached the `max_t` shown by `status`, or a blackout)
`act` and `step` print `Episode is over (...)` and exit 1: stop and summarize.

Next: `grid2op/action_space.md` (every action key), `grid2op/pitfalls.md`
(what breaks silently).
