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
applied redispatch gen_1_0=-5.0 · t=7 · reward=64.1 · new_overloads=[] · illegal=no
```

## It is CUMULATIVE — the #1 redispatch mistake

The delta adds to the gen's running setpoint (`target_dispatch`) and
**persists** on every later step until you counteract it. Verified sequence:
`-5`, `-5`, `+10` → net `-10`, still `-10` after a do-nothing step.

```
# take 5 off gen_1_0, then 5 more  → gen_1_0 is now -10 from base
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
# later: undo it — +10 is legal ONLY after the -5/-5 above (margin then
# allows +10 up); at t=0 with nothing applied, +10 exceeds the ±5 margin
# and the act is ambiguous/no-op. JSON has no "+10" — plain 10.0.
simctl act '{"redispatch": {"gen_1_0": 10.0}}'
```

## Stay inside the margin or the act does nothing

Each gen has headroom this step — `gen_margin_up` / `gen_margin_down` in
`observe --detailed`. At t=0: gen_1_0 **±5**, gen_2_1 **±10**, gen_0_5
**±15**, renewables **0**. A Δ beyond the margin is flagged
`is_ambiguous` → dropped → reward 0. (And redispatching a renewable, e.g.
`gen_5_2`, is unambiguously out of range — margin 0.)

```
# SAFE:  within gen_1_0's ±5 margin at t=0
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
# RISKY: -50 blows the margin → ambiguous, no-op, reward 0
simctl act '{"redispatch": {"gen_1_0": -50.0}}'
```

## Worked relief (verified — and its limit)

`1_4_4` (sub_1→sub_4) hot at 1.28 because sub_1 exports through it. Pulling
the full -5 off gen_1_0 eases it — only from 1.277 to 1.264. That is NOT
enough to stop a trip; redispatch alone won't save a 28%-over line. Use it
for lines in the 0.85–0.95 band, or as one half of a combination with a
topology change (`change_topology.md`), never as the whole fix for an
overloaded line.

```
simctl act '{"redispatch": {"gen_1_0": -5.0}}'
simctl observe        # 1_4_4 rho eases ~1pt; watch sub_1's other feeders
```

## Curtailment (the renewable version)

`curtail` caps a renewable at a fraction of its output — same persistence
mindset:

```
simctl act '{"curtail": {"gen_5_2": 0.5}}'     # wind gen_5_2 to 50%
```
Note the key is **`curtail`**, not `curtailment` (the latter is silently
ignored on this env — see `../docs/grid2op/pitfalls.md` #5).
