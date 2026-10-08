# Recipe: shift generation (redispatch)

**When:** a corridor is hot because a generator is pushing MW into it — or a
renewable is dumping output the grid can't carry. You move generation
between the three dispatchable units: `gen_1_0` (sub_1), `gen_2_1` (sub_2),
`gen_0_5` (sub_0). Renewables can't be redispatched — `curtail` those.

## The command

`redispatch` maps **gen name → ΔMW** (plus = add, minus = take):

```
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
```
```
applied redispatch gen_1_0=-5.0 · t=1 · reward=62.28 · new_overloads=[] · illegal=no
```
In `observe`, the `gens:` line (MW in order gen_1_0, gen_2_1, gen_5_2,
gen_5_3, gen_7_4, gen_0_5) went from `73.2 71.7 36.4 0.0 0.0 69.4` at t=0 to
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
Keep your own running total per gen and undo it when the maneuver is done.

## Stay inside the per-step ramp

Each gen can only change by its ramp per step: gen_1_0 **5 MW**, gen_2_1
**10**, gen_0_5 **15**, renewables **0**. The backend does NOT show margins
(no `gen_margin_*` in `observe` or `--detailed`). A bigger Δ is rejected:
```
$ simctl act '{"redispatch": {"gen_1_0": -50.0}}'
Ambiguous action: Grid2OpException AmbiguousAction InvalidRedispatching "Some redispatching amount are bellow the maximum ramp down"
```
(exit 1, reward 0 for that step, nothing applied — and the step still advances).
Repeated moves within the ramp are fine: -5 five times in a row is accepted.

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
Note the key is **`curtail`**, not `curtailment` (the latter is silently
ignored on this env — see `../docs/grid2op/pitfalls.md` #5).
