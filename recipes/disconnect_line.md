# Recipe: open a line safely

**When:** you need to cut a line's flow — relieve a hot corridor, isolate a
section, or reconfigure before maintenance. Opening a line shifts its MW to
the parallel paths, so the risk is a SECOND line going hot.

(Example from an `l2rpn_case14_sandbox` run; names and numbers differ on
other grids.)

## Step 1 — pick a line that's NOT the only path

Check both endpoint subs have another feeder: `simctl get sub.<name>` lists
every attached line. `0_4_1` (sub_0→sub_4) is a good teaching example:
```
$ simctl get sub.sub_4
...
  attached lines (4):
    0_4_1 ex end, bus=1, status=up, rho=0.2911, p_or=32.191
    1_4_4 ex end, bus=1, status=up, rho=0.6434, p_or=27.709
    3_4_6 ex end, bus=1, status=up, rho=0.4118, p_or=-33.093
    4_5_17 or end, bus=1, status=up, rho=0.448, p_or=18.719
  attached gens (0):
```
(at t=0; the sub's own fields elided.) sub_0 also has `0_1_0`.

## Step 2 — open it and read the result

```
simctl act '{"set_line_status": {"0_4_1": -1}}'
```
```
applied set_line_status 0_4_1=down · t=1 · reward=63.41 · new_overloads=[1_4_4] · illegal=no
```
`new_overloads` names the lines that crossed 1.0 because of your move. One
line per act: opening two lines at once is illegal (`Illegal action: at
most 1 line status change per action (got 2: …)`, the step advances with
reward 0).

## Step 3 — observe, then fix the new hotspot

```
simctl observe
```
```
t=1 reward=63.4 (cum 63.4) done=no
lines_down=1  max_rho=1.02 (1_4_4)  overloads=[1_4_4]
top loads:  1_4_4=101.9%  5_12_9=80.5%  5_10_7=66.0%
gens: 73.3 72.6 36.6 0.0 0.0 73.0
since your last act (t=0 set_line_status 0_4_1=down): no trips; overloads now: [1_4_4] (max_rho=1.02)
```
1_4_4 is hot for the first step (t=1). Doing nothing: still hot at t=2, and
it TRIPS at t=3 (verified) and stays open for its cooldown (10 steps on
case14). You have 2 acts.

Options, in order of preference:

```
# A. re-route sub_4's import off 1_4_4 (see change_topology.md)
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'

# B. undo — a line YOU opened has no cooldown, close it right back
simctl act '{"set_line_status": {"0_4_1": 1}}'
```

After A: `t=2 … lines_down=1 max_rho=0.79 overloads=[]`. After B: `lines_down=0 max_rho=0.80`. Generation relief is weak
here (verified: `redispatch gen_1_0 -5` only moves 1_4_4 from 101.9% to
100.4% — still overloaded, not enough to stop the trip alone).

## Rules of thumb

- One open per turn; let the grid settle; re-observe.
- `overloads` non-empty = the clock is running on that line (trips on its
  3rd consecutive hot step). No "watch it for now".
- If the line you opened was the only path, you'll see `lines_down` climb as
  the isolated section's feeders overload — undo immediately (option B).
