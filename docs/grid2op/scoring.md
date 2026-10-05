# Scoring — reward semantics

Verified on `l2rpn_case14_sandbox`, grid2op 1.12.5.

## The numbers

- Per-step reward range: **[-10.0, 294.53]** (`env.reward_range`).
- A healthy, do-nothing step pays **≈ 64–65**. You'll see 63–66 most of the
  time when the grid is calm.
- The reward is **loss-based**: you start from a base and get charged for
  overloads (lines above `rho=1.0`), losses, and especially trips.
- **Higher is better.** `cum_reward` is your episode score — the sum of step
  rewards. `simctl status` shows both:
  `reward=63.1 (cum 3120.4)`.
- A step where your action was **illegal or ambiguous pays 0** (the action
  is dropped, the grid steps as do-nothing). That's an instant -64 vs doing
  nothing, so rejected acts are expensive — read the reason and fix the move.
- A tripped line costs more than one step: the grid runs degraded (loads
  re-routed, siblings hot) while it's out (~10-step cooldown + whatever
  overloads the re-route caused). Cascades — a trip that trips a sibling —
  are where reward collapses toward -10 and the episode can end early.

## What "good" looks like

- `max_rho` under ~0.9. Lines at 0.95+ are one load swing away from tripping;
  renewables move every step, so "hot but under 1.0" is a ticking clock, not
  a steady state.
- `overloads=[]` and `lines_down=0` on every `observe`.
- No trips since your last act — the feedback line says so:
  `since your last act (…): 2_3_5 +9.0%, no trips`. If it names a trip
  instead, you caused it; undo or re-route NOW, before the cooldown and the
  cascade.
- Cumulative reward climbing at ~64/step.

## Reading the feedback line

`simctl observe` ends with:

```
since your last act (t=51 set_line_status 0_4_1=-1): 2_3_5 +9.0%, no trips
```

= "since the step you acted on, line 2_3_5's loading rose 9.0 percentage
points and nothing tripped." This is the single most information-dense line
in the output: it closes the loop between your decision and the physical
consequence. If it shows a big jump or a trip, your next act should address
that line, not start a new maneuver.

## Episode end

When `done=yes` (cascade, or t=8064 reached), `simctl status` shows
`done=yes` and the final `cum_reward` is the episode score. The observation
at `done` is a game-over state — don't parse it. `simctl reset` for the next
episode.
