# Action space — every key, exact JSON

You act with `simctl act '<json>'`. The JSON is a grid2op 1.12.4 action
dict. All shapes below were verified against the installed library on
`l2rpn_case14_sandbox`. **Acting also advances one timestep** (grid2op
semantics: a step applies the action, then runs the grid forward 5 minutes).

Element names are exact strings from the env: lines like `0_4_1` (see
`env.name_line` order in `environment.md`), gens `gen_1_0`, loads `load_1_0`,
subs `sub_0`.

## The keys (what works on this env)

### `set_line_status` — force a line's state
Values: `1` = force closed, `-1` = force open, `0` = no-op.
```
simctl act '{"set_line_status": {"0_4_1": -1}}'     # open 0_4_1
simctl act '{"set_line_status": {"1_4_4": 1}}'      # force-close 1_4_4 (ILLEGAL while a tripped line is in cooldown)
simctl act '{"set_line_status": {"0_1_0": 0}}'      # explicit no-op on 0_1_0
```
Opening a line shifts its flow to parallel paths — check `new_overloads`.
Closing a line you opened yourself is instant (no cooldown).
Closing a line that TRIPPED is illegal until its cooldown runs out.

### `change_line_status` — toggle a line (up→down, down→up)
**Takes a LIST of line names/ids, not a name→bool dict.** The dict form
`{"0_1_0": true}` raises `AmbiguousAction` in 1.12.4 (the older grid2op docs
show the dict form — ignore them).
```
simctl act '{"change_line_status": ["0_4_1"]}'      # toggle 0_4_1
```
One line per act: at most ONE line's status may change per act
(`set_line_status` too). `["0_4_1", "1_4_4"]` prints `Illegal action: … More
than 1 line status affected by the action` — the step advances, reward 0.
Use this when you want "switch it" without deciding which way; use
`set_line_status` when you want a guaranteed state.

### `set_bus` — force an element onto a specific busbar (1 or 2)
Sub-keys: `lines_or_id`, `lines_ex_id` (line ends), `loads_id`,
`generators_id`. Each maps **name → bus number 1|2**.
```
simctl act '{"set_bus": {"lines_or_id": {"0_4_1": 2}}}'    # 0_4_1's or end → sub_0 busbar 2
simctl act '{"set_bus": {"lines_ex_id": {"3_4_6": 1}}}'    # 3_4_6's ex end → sub_4 busbar 1
simctl act '{"set_bus": {"generators_id": {"gen_1_0": 2}}}'
```
> set_bus docstring: "Setting a bus has the effect to assign the object to
> this bus. If it was before that connected to bus 1, and you assign it to
> bus 1 it will stay on bus 1. If it was on bus 2 (and you still assign it to
> bus 1) it will be moved to bus 1."

Busbar 1 is the main busbar; busbar 2 the backup. Moving a line END onto
busbar 2 disconnects that end from the sub's main flow **without opening the
line** — power re-routes through the sub's sibling lines. Only `1` or `2`
are valid (this env has 2 busbars per sub); bus `3` raises `AmbiguousAction`.

### `change_bus` — toggle an element to the other busbar (1→2, 2→1)
Sub-keys: `lines_or_id`, `lines_ex_id`, `loads_id`, `generators_id`. Each maps
**name → true**, or just give the LIST of names:
```
simctl act '{"change_bus": {"lines_or_id": ["1_4_4"]}}'    # 1_4_4's or end flips busbar
simctl act '{"change_bus": {"lines_ex_id": ["0_4_1"]}}'
simctl act '{"change_bus": {"loads_id": ["load_3_2"]}}'    # move a load to the other busbar
```
> change_bus docstring: "Changing a bus has the effect to assign the object
> to bus 1 if it was before that connected to bus 2, and to assign it to bus
> 2 if it was connected to bus 1."

Verified re-route (from `quickstart.md`): after opening 0_4_1,
`change_bus lines_or_id ["1_4_4"]` takes 1_4_4 off the main flow and (chronic
0) max_rho falls to 0.79 — the load you moved landed on other lines, which
may heat up in other scenarios. Always re-observe after a bus change.

### `redispatch` — shift a dispatchable generator by ΔMW
Maps **gen name → ΔMW** (or a list of `[id, ΔMW]` pairs). Only the 3
dispatchable gens respond: `gen_1_0` (sub_1), `gen_2_1` (sub_2), `gen_0_5`
(sub_0). Renewables can't be redispatched.
```
simctl act '{"redispatch": {"gen_1_0": -5.0}}'    # take 5 MW off gen_1_0
simctl act '{"redispatch": {"gen_2_1": 10.0}}'    # add 10 MW to gen_2_1
```
**CUMULATIVE and PERSISTENT.** The delta adds to the generator's running
setpoint (grid2op keeps `target_dispatch`); it survives subsequent no-op
steps until you counteract it. -5 then -5 = -10. To undo, send +5 (or +10).
Keep |Δ| within the gen's per-step ramp: gen_1_0 ±5, gen_2_1 ±10, gen_0_5
±15. Margins are not shown by `observe`/`--detailed`; a larger Δ (e.g. `-50`)
is rejected as `Ambiguous action` — reward 0, nothing applied, step still
advances. Keep your own running total.

### `curtail` — cap a renewable generator
Maps **gen name → fraction 0..1** of its max output.
```
simctl act '{"curtail": {"gen_5_2": 0.5}}'        # wind gen_5_2 capped at 50%
```
The action echoes as: `Limit unit "gen_5_2" to 50.0% of its Pmax`. Use it to
dump unwanted renewable output that's pushing a line hot. (Note: the key is
`curtail` — see "keys that silently no-op" below.)

### Combining keys

One `act` can carry several keys — they apply together, then one step runs:
```
simctl act '{"set_line_status": {"0_4_1": -1}, "redispatch": {"gen_1_0": -5.0}}'
```
Empty `{}` is the do-nothing action (equivalent to `simctl step`).

## Keys that SILENTLY NO-OP on this env (grid2op 1.12.4)

The env's action space (`PlayableAction`) accepts only: `set_line_status`,
`change_line_status`, `set_bus`, `change_bus`, `redispatch`, `set_storage`,
`curtail`, `raise_alarm`, `raise_alert`. Any OTHER key in your JSON is
**ignored with a warning** — your act "succeeds", the grid steps, and nothing
changed:

- `curtailment` → use `curtail`
- `curtail_mw` → not here
- `detach_load` / `attach_load` → not enabled in this env
- `set_storage_power` → not here (no storage anyway; `set_storage` raises
  `IllegalAction`)
- **typos** — `set_line_statu` is silently dropped (the output still says
  `applied set_line_statu …`). If an act seems to do
  nothing and isn't marked illegal/ambiguous, re-check your key spelling
  against this page.

A nonexistent element NAME or a wrong value shape, by contrast, is loud:
building the action raises `AmbiguousAction` ("No known line with name …").
The act is rejected (exit 1, `Illegal — <reason>` on stdout) and **time does
NOT advance**.

## The factory methods (for reference)

The backend can also build actions via `env.action_space`'s factory methods —
real signatures from the installed lib:

```python
change_bus(name_element: str, extremity: Literal['or','ex'] = None,
           substation: int = None, type_element: str = None,
           previous_action: BaseAction = None) -> BaseAction
set_bus(name_element: str, new_bus: int, extremity: Literal['or','ex'] = None,
        substation: int = None, type_element: int = None,
        previous_action: BaseAction = None) -> BaseAction
disconnect_powerline(line_id: int = None, line_name: str = None,
                     previous_action: BaseAction = None) -> BaseAction
reconnect_powerline(bus_or: int, bus_ex: int, line_id: int = None,
                    line_name: str = None, previous_action: BaseAction = None)
    -> BaseAction
```
`extremity` is `"or"` or `"ex"` for lines (ignored for gens/loads).
`previous_action` chains moves into one action (it is modified **in place** —
the docstrings warn about this). `reconnect_powerline` bus args: pass `0` to
mean "last known bus". You won't call these yourself — `simctl act` JSON is
your interface — but the semantics above match them 1:1.

## Legal / illegal / ambiguous

| | happens when | what you see | grid effect |
|---|---|---|---|
| **legal** | normal | `applied … illegal=no`, exit 0 | action + 1 step |
| **illegal** | not allowed in this state (close a line mid-cooldown, change 2 lines at once) | `Illegal action: <reason>`, exit 1 | 1 step as do-nothing, **reward 0** |
| **malformed** | can't be built (bus>2, name→bool where a list is required, unknown element name) | `Illegal — <reason>`, exit 1 | **no step**, time does not advance |

From `Environment.step`'s docstring: "If the action is illegal or
ambiguous, the step is performed, but the action is replaced with a 'do
nothing' action." (An `Ambiguous action: <reason>` message, same effect as
illegal, is possible too.) Read the reason, adjust, re-act.

## Cooldowns (verified numbers)

- **Overload trip**: a line above `rho=1.0` trips on its **3rd consecutive
  overloaded step** (2 allowed, verified; hard overflow at `rho≥2.0` is
  instant). A tripped line is OPEN, `status: "down"`, `cooldown: 10`
  counting down — closing it before 0 is illegal.
- **Self-opened line**: no re-close cooldown. `set_line_status … 1` works
  immediately.
- `observe --detailed` shows `cooldown: <steps left>` per line (and
  `status: "cooldown"` for a connected line still in cooldown).

## Which keys shift load where (cheat sheet)

| you do | load moves to |
|---|---|
| open line `A` | parallel paths between A's two subs (siblings in the ring) |
| `change_bus` a line end | the sub's OTHER lines feeding that side (busbar 1 ↔ 2 split) |
| `set_bus` a line end to 2 | same, forced: that end is off the main busbar |
| `redispatch` gen −Δ | neighboring subs take the MW (via their feeders) |
| `curtail` a renewable | neighboring subs / other gens absorb the dump |

Every one of these is a re-routing, not a deletion: the MW goes somewhere.
The line that takes it is your new risk. Observe after every act.
