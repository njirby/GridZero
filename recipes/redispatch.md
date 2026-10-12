# Recipe: shift generation (redispatch)

**When:** a corridor is hot because a generator is pushing MW into it — or a
renewable is dumping output the grid can't carry. You move generation
between the dispatchable units (`redispatchable: true`; the list with ramps
is in "Grid facts" in AGENTS.md). Renewables can't be redispatched —
`curtail` those. (Examples below are from an `l2rpn_case14_sandbox` run,
whose dispatchable gens are `gen_1_0`, `gen_2_1`, `gen_0_5`; names and
numbers differ on other grids.)

## The command

`redispatch` maps **gen name → ΔMW** (plus = add, minus = take):

```
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
```
```
applied redispatch gen_1_0=-5.0 · t=1 · reward=62.28 · new_overloads=[] · illegal=no
```
In `observe`, the `gens:` line (MW in gen id order; on case14 gen_1_0,
gen_2_1, gen_5_2, gen_5_3, gen_7_4, gen_0_5) went from `73.2 71.7 36.4 0.0 0.0 69.4` at t=0 to
`68.3 74.6 36.6 0.0 0.0 74.5` at t=1: gen_1_0 dropped 5 MW and the other
dispatchable units picked up the balance.

## It is CUMULATIVE — the #1 redispatch mistake

The delta adds to the gen's running setpoint (`target_dispatch`) and
**persists** on every later step until you counteract it:

```
# take 5 off gen_1_0, then 5 more  → gen_1_0 is now -10 from base
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
# later: undo it with the opposite sign, again ≤5 per step (plain 5.0, JSON has no "+5")
simctl act '{"redispatch": {"gen_1_0": 5.0}}'
simctl act '{"redispatch": {"gen_1_0": 5.0}}'
```
The running total per gen is shown as a flag on its `grid` / `list gens`
row (`target` = what you asked for, `actual` = what the grid delivers; the
other dispatchable gens move to rebalance):
```
$ simctl list gens --where redispatchable=true
3 of 3 matching gens (of 6) · where redispatchable=true — name sub p(MW) R=redispatchable W=renewable [flags: …]
  gen_1_0 sub_1      67.9 R- dispatch target=-5 actual=-4.5
  gen_2_1 sub_2      71.8 R- dispatch target=0 actual=1.75
  gen_0_5 sub_0      76.5 R- dispatch target=0 actual=2.75
```
`simctl history gen.gen_1_0` shows `p target_dispatch actual_dispatch` per
step. Undo it when the maneuver is done.

## Stay inside the per-step ramp

Each gen can only change by its ramp per step (listed in "Grid facts"; on
case14 gen_1_0 **5 MW**, gen_2_1 **10**, gen_0_5 **15**), renewables **0**.
The backend does NOT show margins
(no `gen_margin_*` in any view). A bigger Δ is rejected:
```
$ simctl act '{"redispatch": {"gen_1_0": -50.0}}'
Ambiguous action: redispatch gen_1_0 -50 MW exceeds its ramp-down limit 5 MW/step (current target_dispatch 0 MW). Not applied: the step advanced to t=3 with do-nothing instead (reward 0).
```
(exit 1, reward 0 for that step, nothing applied — and the step still advances).
Repeated moves within the ramp are fine: -5 five times in a row is accepted,
up to a cumulative `target_dispatch` of the gen's `pmax - pmin` (beyond that
it is reported as illegal and not applied).

## Worked relief (verified — and its limit)

`1_4_4` (sub_1→sub_4) at 101.9% after opening `0_4_1`. Pulling the full -5
off gen_1_0 eases it — only to 100.4%, still over 1.0. Redispatch alone
won't save a hot line in the 100%+ range. Use it for lines in the 0.85–0.95
band, or as one half of a combination with a topology change
(`change_topology.md`).

```
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
simctl observe        # 1_4_4 rho eases ~1-2 points
```

## Curtailment (the renewable version)

`curtail` caps a renewable at a fraction of its output — same persistence
mindset:

```
simctl act '{"curtail": {"gen_5_2": 0.5}}'     # gen_5_2 capped at 50%
```
```
applied curtail gen_5_2=0.5 · t=3 · reward=64.14 · new_overloads=[] · illegal=no
```
The value is a fraction 0..1; anything else is rejected without advancing
time (`Invalid action: curtail value for gen_5_2 must be between 0.0 and 1.0
(fraction of pmax), got 1.5. …`).
Only gens listed as curtailable in "Grid facts" take it. Note the key is
**`curtail`**, not `curtailment` (the latter is silently
ignored on this env — see `../docs/grid2op/pitfalls.md` #5).
