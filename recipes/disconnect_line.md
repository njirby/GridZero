# Recipe: open a line safely

**When:** you need to cut a line's flow — relieve a hot corridor, isolate a
section, or reconfigure before maintenance. Opening a line shifts its MW to
the parallel paths, so the risk is a SECOND line going hot.

## Step 1 — pick a line that's NOT the only path

From `observe --detailed`, check both endpoint subs have another feeder.
`0_4_1` (sub_0→sub_4) is a good teaching example: sub_4 also has `1_4_4`
(from sub_1) and `3_4_6` (from sub_3); sub_0 has `0_1_0`.

## Step 2 — open it and read the result

```
simctl act '{"set_line_status": {"0_4_1": -1}}'
```
```
applied set_line_status 0_4_1=down · t=1 · reward=63.41 · new_overloads=[1_4_4] · illegal=no
```
`new_overloads` names the lines that crossed 1.0 because of your move. One
line per act: opening two lines at once is illegal (`More than 1 line
status affected`, the step advances with reward 0).

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
it TRIPS at t=3 (verified) and stays open 10 steps. You have 2 acts.

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
