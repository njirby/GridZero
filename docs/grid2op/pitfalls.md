# Pitfalls — grid2op 1.12.5 gotchas

Everything here bit someone or was verified live. Read this after any
rejected act; keep it in your head when composing JSON.

## API shape (1.12 "glop" restructure)

1. **The env has no grid object.** Pre-1.12 tutorials show the env exposing
   a grid attribute with `.rho` / `.line_status` — it does not exist in
   1.12 (the "glop" restructure removed it). Names come from
   `env.name_line` / `env.name_sub` / `env.name_gen` / `env.name_load`;
   limits from `env.get_thermal_limit()`. You drive the grid via `simctl`
   anyway — just don't write code assuming the old API.
2. **`obs.line_status` is a BOOL array** — `True` = connected, `False` =
   down. The old API used ints (1/-1). The C3 JSON you read via
   `observe --detailed` renders it as `status: "up"/"down"/"cooldown"/"maintenance"`.
3. **`set_line_status` values are `1` / `-1` / `0`**, not booleans.
   `1` = force closed, `-1` = force open, `0` = no-op. `true`/`false` are
   not the values the parser wants.
4. **`change_line_status` takes a LIST, not a name→bool dict.**
   `{"change_line_status": ["0_1_0"]}` works; `{"change_line_status":
   {"0_1_0": true}}` raises `AmbiguousAction` in 1.12.5. Same for
   `change_bus` sub-keys: `{"change_bus": {"lines_or_id": ["1_4_4"]}}` is
   right, `{"lines_or_id": {"1_4_4": true}}` is not. (`set_bus`, by contrast,
   IS a name→bus-number dict: `{"set_bus": {"lines_or_id": {"0_4_1": 2}}}`.)

## Silent failures (the dangerous kind)

5. **Unknown action keys are IGNORED, with only a python warning.** Your act
   "succeeds", the grid steps, nothing changed, and no error surfaces. Keys
   that silently no-op on this env: `curtailment` (use `curtail`),
   `curtail_mw`, `detach_load`, `attach_load`, `set_storage_power`.
   **Typos are in the same category**: `set_line_statu` is dropped. If an act
   did nothing and wasn't marked illegal/ambiguous, re-check key spelling
   against `action_space.md`.
6. **`rho` is NOT `p_or / limit`.** It's current-based:
   `max(|a_or|, |a_ex|) / limit`, and `env.get_thermal_limit()` returns AMPS
   (541, 450, 375, …) while `p_or`/`p_ex` are MW. A line can have `p_or = 0`
   and `rho = 0.43` (reactive flow, e.g. 6_7_18 at t=0). Just read `rho` —
   don't recompute it.

## Physics traps

7. **Don't open a line that's the only path.** Most subs have 2–4 feeders
   (sub_4 has four: 0_4_1, 1_4_4, 3_4_6, 4_5_17), so most opens re-route
   fine — but the re-route lands somewhere.
   Verified at t=0: opening `0_4_1` pushes `1_4_4` to rho 1.277 in ONE step.
   If a substation is left with a single feeder, any further open isolates
   it → cascade. Check `new_overloads` before celebrating an open.
8. **A line hot for 2 consecutive steps trips** (soft-overflow threshold
   1.0, allowed 2 timesteps; rho ≥ 2.0 trips instantly). So a line at
   1.05 on `observe` is NOT a maybe — it trips next step unless you act
   this step.
9. **Tripped lines have a ~10-step reconnection cooldown.** Attempting
   `set_line_status … 1` on a tripped line during cooldown is ILLEGAL:
   the step pays 0 and the line stays down (verified: reward 0.0, `illegal`,
   cooldown ticking 10 → 9). A line YOU opened yourself has NO cooldown —
   you can close it the very next step. Know which kind of "down" you're
   dealing with (`status` field: `down` = your open, `cooldown` = tripped).
10. **`redispatch` is cumulative.** -5, then -5, then +10 = net -10, and it
    persists across no-op steps (`target_dispatch` stays -10). Undo your own
    dispatch when the maneuver is done. Redispatch beyond a gen's margin
    (`gen_margin_up/down`) → flagged `is_ambiguous` → no-op + reward 0.
    Margins at t=0: gen_1_0 ±5, gen_2_1 ±10, gen_0_5 ±15; renewables 0
    (can't be redispatched — `curtail` them instead).
11. **Bus changes re-route, they don't delete.** Moving a line end to
    busbar 2 takes it off the sub's main flow; the MW re-enters through the
    sub's sibling lines. Verified: isolating `1_4_4`'s or end after opening
    `0_4_1` drops 1_4_4 to 0 but pushes `4_5_17` to 0.908. Always re-observe
    after a bus move.
12. **Only busbars 1 and 2 exist.** `set_bus` to `3` raises
    `AmbiguousAction` at build time.

## Process traps

13. **A rejected act still advances time.** grid2op replaces illegal/
    ambiguous actions with do-nothing and steps anyway. The chronic moved
    while you were fixing your JSON; re-observe before re-acting.
14. **`done` observations are garbage.** After the episode ends, the last
    observation is a "game over" state (per `Environment.step`'s docstring:
    "the observation is NOT properly updated and should not be used at all").
    Read the final `cum_reward` from `status`, then `reset`.
15. **`render` costs ~750 ms and ~676 tokens to look at.** It's for topology
    questions ("which line connects sub_4 to sub_1?"), not for every turn.
    `observe` text answers 95% of what you need.
