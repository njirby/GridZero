# Recipe: a complete turn (observe → decide → act → verify → repeat)

One full operator turn with real sandbox numbers at t=0. Copy this rhythm;
vary the decision.

**1. Read the state.**
```
simctl observe
```
```
t=0 reward=64.99 (cum 64.99) done=no
lines_down=0  max_rho=0.92 (4_5_17)  overloads=[]
top loads:  4_5_17=92.4%  1_4_4=82.8%  5_12_9=73.2%  3_6_15=54.5%
gens: 81.4 79.3 5.3 0.0 0.0 82.2 | loads: 5.4 12.6 14.4 ...
since your last act: (none yet this episode)
```
Interpretation: nothing above 1.0, but `4_5_17` (sub_4→sub_5, the transit
line for the lower ring) sits at 92% — the warmest line on the grid.
`1_4_4` (sub_1→sub_4) is second at 83%.

**2. Decide, out loud, in 1–2 sentences.**
"4_5_17 is hot at 92% but under 1.0 — not an emergency. I'll leave topology
alone and make one small verified move: pull 5 MW off gen_1_0, which cools
the sub_1→sub_4 corridor (1_4_4). 5 MW is within gen_1_0's ±5 margin, so
the act can't be rejected."

(Self-correction habit: name the second-order effect BEFORE acting. In this
grid, pulling the upper gens does NOT cool 4_5_17 — it's fed by lower-ring
transit. Know which lines your lever actually touches.)

**3. Act.**
```
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
```
```
applied redispatch gen_1_0=-5.0 · t=1 · reward=63.8 · new_overloads=[] · illegal=no
```
Check the tail: `new_overloads=[]`, `illegal=no`. If either fails, stop —
read the reason, don't stack more moves.

**4. Verify.**
```
simctl observe
```
```
t=1 reward=63.8 (cum 128.8) done=no
lines_down=0  max_rho=0.93 (4_5_17)  overloads=[]
top loads:  4_5_17=92.5%  1_4_4=81.7%  5_12_9=73.9%  3_6_15=54.4%
...
since your last act (t=0 redispatch gen_1_0=-5.0): 1_4_4 -1.1%, no trips
```
The feedback line confirms the intent: 1_4_4 cooled, nothing tripped, no
new overloads. 4_5_17 is unchanged (as predicted) — it's still your watch
item, not a new problem you created.

**5. Repeat — but bookkeep.**
- The -5 MW on gen_1_0 is **still applied** (redispatch is cumulative;
  `target_dispatch` stays -5.0 through no-op steps). When the maneuver is
  done, undo it: `simctl act '{"redispatch": {"gen_1_0": 5.0}}'`.
- If 4_5_17 crosses 1.0, the relief is structural (split the lower ring's
  import), not more redispatch — see `change_topology.md`.
- Every 20–30 turns, or after any structural move, `simctl render` + `Read`
  re-grounds you in the topology you've built.

## When a turn goes wrong

- `illegal=yes` / `ambiguous=yes` → the reason is in the act output; your
  move was dropped but time advanced. Re-observe, fix, re-act.
- `new_overloads=[X]` → X crossed 1.0 this step. Your NEXT act must cool X
  (re-route, redispatch, or undo) before anything else.
- `overloads=[X]` persists in two consecutive observes → X trips next step.
  If you can't relieve it, undo whatever heated it (your own opens have no
  cooldown — close them instantly).
