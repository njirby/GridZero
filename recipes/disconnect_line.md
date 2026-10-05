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
applied set_line_status 0_4_1=down · t=1 · reward=64.0 · new_overloads=[1_4_4] · illegal=no
```
`new_overloads` names the lines that crossed 1.0 because of your move.
At t=0 on the sandbox, opening 0_4_1 takes `1_4_4` to **rho 1.277 in one
step** — it trips next step if you do nothing.

## Step 3 — observe, then fix the new hotspot

```
simctl observe
```
```
t=1 reward=64.0 (cum 128.5) done=no
lines_down=1  max_rho=1.28 (1_4_4)  overloads=[1_4_4]
top loads:  1_4_4=128.0%  4_5_17=82.0%  0_1_0=58.0%
...
since your last act (t=0 set_line_status 0_4_1=-1): 1_4_4 +53.0%, no trips
```

Options, in order of preference:

```
# A. re-route sub_4's import off 1_4_4 (see change_topology.md)
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'

# B. undo — a line YOU opened has no cooldown, close it right back
simctl act '{"set_line_status": {"0_4_1": 1}}'
```

After B at t=1 on the sandbox, `1_4_4` relaxes to rho ≈ 0.83. Generation
relief is weak here (verified: `redispatch gen_1_0 -5` only moves 1_4_4
from 1.277 to 1.264 — not enough to stop the trip). If you chose A, note
that 4_5_17 climbs to ~0.91 and becomes the new watch item — see
`change_topology.md` for why you don't chase it with more bus moves.

## Rules of thumb

- One open per turn; let the grid settle; re-observe.
- `overloads` non-empty = that line trips NEXT step. No "watch it for now".
- If the line you opened was the only path, you'll see `lines_down` climb as
  the isolated section's feeders overload — undo immediately (option C).
