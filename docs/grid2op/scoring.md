# Scoring — reward semantics

Verified on `l2rpn_case14_sandbox`, grid2op 1.12.4. The numbers below are
case14's; the reward scale (range, a healthy step) of your grid is in the
"Grid facts" section of AGENTS.md.

## The numbers

- Per-step reward range on case14: **[-10.0, 294.53]** (`env.reward_range`).
- A healthy, do-nothing step on case14 pays **≈ 64–65** (63–66 most of the
  time when the grid is calm). Learn your grid's healthy level from "Grid
  facts" and your first few steps.
- The reward is **loss-based**: you start from a base and get charged for
  overloads (lines above `rho=1.0`), losses, and especially trips.
- **Higher is better.** `cum_reward` is your episode score — the sum of step
  rewards. `simctl status` shows both:
  `reward=63.1 (cum 3120.4)`. At `t=0` it shows `reward=-10.0 (cum 0.0)`: a
  placeholder from the reset before any step; it is not part of `cum`.
- A step where your action was **illegal or ambiguous pays 0** (the action
  is dropped, the grid steps as do-nothing). That's a full healthy step's
  reward lost vs doing nothing, so rejected acts are expensive — read the reason and fix the move.
- A tripped line costs more than one step: the grid runs degraded (loads
  re-routed, siblings hot) while it's out (its cooldown + whatever
  overloads the re-route caused). Cascades — a trip that trips a sibling —
  are where reward collapses toward -10 and the episode can end early.

## What "good" looks like

- `max_rho` under ~0.9. Lines at 0.95+ are one load swing away from tripping;
  renewables move every step, so "hot but under 1.0" is a ticking clock, not
  a steady state.
- `overloads=[]` and `lines_down=0` on every `observe`.
- No trips since your last act — the feedback line says so:
  `since your last act (…): no trips`. If it says `tripped: [...]` instead
  (or `lines_down` went up), a line is out for its cooldown; re-route NOW,
  before the cascade.
- Cumulative reward climbing by about a healthy step each step.

## Reading the feedback line

`simctl observe` ends (after your first act) with:

```
since your last act (t=0 set_line_status 0_4_1=down): no trips; overloads now: [1_4_4] (max_rho=1.01)
```

= "the step after your act at t=0: nothing tripped, but 1_4_4 is now above
1.0 and the hottest line is at 1.01." It is the shortest route from your
decision to its physical consequence. On a trip it reads
`tripped: [<line>]` instead of `no trips`. If it shows overloads or a trip,
your next act should address that line, not start a new maneuver.

## Episode end

When `done=yes` (cascade, or the episode ran its full length), `simctl
status` shows `done=yes` and the final `cum_reward` is the episode score.
Further `act`/`step` calls exit 1 with `Episode is over (...)`. Report the
final `cum` and stop; you cannot start another episode.
