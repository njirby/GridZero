# Recipe: a complete turn (observe → decide → act → verify → repeat)

One full operator turn with real sandbox numbers at t=0 (chronic 0). Copy
this rhythm; vary the decision.

**1. Read the state.**
```
simctl observe
```
```
t=0 reward=-10.0 (cum 0.0) done=no
lines_down=0  max_rho=0.80 (5_12_9)  overloads=[]
top loads:  5_12_9=79.9%  5_10_7=66.7%  1_4_4=64.3%
gens: 73.2 71.7 36.4 0.0 0.0 69.4
```
No "since your last act" line yet — you haven't acted. `reward=-10.0` is the
t=0 placeholder. Interpretation: nothing above 1.0; `5_12_9` is the warmest
line at 80%.

**2. Decide, out loud, in 1–2 sentences.**
"Nothing is overloaded and max_rho is 0.80, so no emergency. I'll make one
small verified move: pull 5 MW off gen_1_0, which is within its 5 MW per-step
ramp, and watch where the MW goes."

(Self-correction habit: name the second-order effect BEFORE acting. Know
which lines your lever actually touches.)

**3. Act.**
```
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
```
```
applied redispatch gen_1_0=-5.0 · t=1 · reward=62.28 · new_overloads=[] · illegal=no
```
Check the tail: `new_overloads=[]`, `illegal=no`. If either fails, stop — read
the reason, don't stack more moves.

**4. Verify.**
```
simctl observe
```
```
t=1 reward=62.3 (cum 62.3) done=no
lines_down=0  max_rho=0.81 (5_12_9)  overloads=[]
top loads:  5_12_9=80.7%  5_10_7=66.5%  1_4_4=63.8%
gens: 68.3 74.6 36.6 0.0 0.0 74.5
since your last act (t=0 redispatch gen_1_0=-5.0): no trips (max_rho=0.81)
```
The feedback line confirms nothing tripped; `gens:` shows gen_1_0 down ~5 MW
with gen_2_1 and gen_0_5 picking up. 1_4_4 eased 64.3% → 63.8%.

**5. Repeat — but bookkeep.**
- The -5 MW on gen_1_0 is **still applied** (redispatch is cumulative; it
  persists through no-op steps). When the maneuver is done, undo it:
  `simctl act '{"redispatch": {"gen_1_0": 5.0}}'`.
- If a line crosses 1.0, the relief is usually structural (open/re-route,
  see `disconnect_line.md`, `change_topology.md`), not more redispatch.
- Stop when `done=yes` (the horizon in `simctl status`, e.g. `t=24/24`).

## When a turn goes wrong

- `Illegal action: …` (exit 1) → your move was dropped but time advanced
  and the step paid 0. Re-observe, fix, re-act. (`Illegal — …` means the
  action couldn't even be built; time did NOT advance.)
- `new_overloads=[X]` → X crossed 1.0 this step. Your NEXT act must cool X
  (re-route, redispatch, or undo) before anything else.
- `overloads=[X]` still present at the next observe → X is on its 2nd hot
  step; the following step trips it. If you can't relieve it, undo whatever
  heated it (your own opens have no cooldown — close them instantly).
