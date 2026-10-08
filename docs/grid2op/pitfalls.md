# Pitfalls — grid2op 1.12.4 gotchas

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
   {"0_1_0": true}}` raises `AmbiguousAction` in 1.12.4. Same for
   `change_bus` sub-keys: `{"change_bus": {"lines_or_id": ["1_4_4"]}}` is
   right, `{"lines_or_id": {"1_4_4": true}}` is not. (`set_bus`, by contrast,
   IS a name→bus-number dict: `{"set_bus": {"lines_or_id": {"0_4_1": 2}}}`.)

## Silent failures (the dangerous kind)

5. **Unknown action keys are IGNORED, with only a python warning.** Your act
   "succeeds", the grid steps, nothing changed, and no error surfaces (`simctl act` even prints
   `applied <key> …` and `illegal=no`). Keys
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
   Verified at t=0 (chronic 0): opening `0_4_1` pushes `1_4_4` to rho 1.019
   in ONE step (other scenarios push it much higher).
   If a substation is left with a single feeder, any further open isolates
   it → cascade. Check `new_overloads` before celebrating an open.
8. **A line trips on its 3rd consecutive step above 1.0** (verified on
   grid2op 1.12.4, `NB_TIMESTEP_OVERFLOW_ALLOWED=2`: hot at t=2 counter 1,
   t=3 counter 2, gone at t=4). The first observe that shows `overloads=[X]`
   is step 1: you have 2 acts, and the act you take at the second hot
   observe is the last one that can save it. A line at 1.05 is NOT a maybe.
   (Loading ≥ 2.0 is a hard overflow and trips instantly.)
9. **Tripped lines are OPEN for 10 steps.** A tripped line is
   disconnected (`status: "down"`, `cooldown: 10` counting down). Attempting
   `set_line_status … 1` on it during cooldown is ILLEGAL: the step pays 0
   and the line stays down (verified: `Illegal action: … cooldown of [10]`,
   reward 0.0, cooldown ticking 10 → 9). A line YOU opened yourself has NO
   cooldown — you can close it the very next step. The `status` field says
   `down` for both; `cooldown > 0` marks a tripped one.
10. **`redispatch` is cumulative.** -5, then -5, then +10 = net -10, and it
    persists across no-op steps (`target_dispatch` stays -10). Undo your own
    dispatch when the maneuver is done. Per-step ramps: gen_1_0 5 MW, gen_2_1
    10, gen_0_5 15; renewables 0 (`curtail` them instead). The backend does
    NOT show margins; a larger single move (e.g. `-50`) is `Ambiguous action`
    (reward 0, nothing applied), so stay within the ramp and keep your own
    running total per gen.
11. **Bus changes re-route, they don't delete.** Moving a line end to
    busbar 2 takes it off the sub's main flow; the MW re-enters through the
    sub's sibling lines. Verified: moving `1_4_4`'s or end to busbar 2 after opening
    `0_4_1` drops 1_4_4 to 0 and (chronic 0) leaves max_rho at 0.79; in other
    scenarios a sibling can heat up instead. Always re-observe after a bus
    move.
12. **Only busbars 1 and 2 exist.** `set_bus` to `3` raises
    `AmbiguousAction` at build time.

## Process traps

13. **A rejected act usually still advances time.** `Illegal action: …`
    (exit 1): grid2op replaced it with do-nothing and stepped anyway, reward 0.
    The chronic moved while you were fixing your JSON; re-observe before
    re-acting. The exception: an action that can't be built at all (wrong
    shape, unknown element name, bus 3) prints `Illegal — …`, exit 1, and
    time does NOT advance.
14. **After `done`, stop.** The episode can't be reset. `act`/`step` print
    `Episode is over (<cause>). Stop acting and write your summary.` (exit 1).
    Read the final `cum_reward` from `status` and summarize.
15. **`render` may be disabled and is rarely needed.** Where it works it costs
    tokens to look at; where it doesn't it prints `render is disabled in this
    environment` (exit 1) — don't retry. `--out` takes a bare filename only.
    `observe --detailed` has each line's `or`/`ex` substations (topology).
