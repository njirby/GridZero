# Recipe: check the state

**When:** start of your session, after any act, whenever unsure. This is the
read-only pair — it never changes the grid.

```
simctl status
```
```
sim up · env=l2rpn_case14_sandbox · t=0/24 · reward=-10.0 (cum 0.0) · done=no
```
`t=0/24` = step 0 of a 24-step horizon; the episode ends at `max_t`. The
`-10.0` at t=0 is a placeholder before any step. `done=no` = running.
`sim down` or a backend error (exit 2) → retry `simctl status`; you cannot
reset the episode — if the sim stays down, the run is over. After
`done=yes`, stop and write your summary.

```
simctl observe
```
```
t=2 reward=64.0 (cum 127.4) done=no
lines_down=1  max_rho=1.01 (1_4_4)  overloads=[1_4_4]
top loads:  1_4_4=101.2%  5_12_9=80.0%  5_10_7=65.7%
gens: 73.9 72.7 35.6 0.0 0.0 71.0
since your last act (t=0 set_line_status 0_4_1=down): no trips; overloads now: [1_4_4] (max_rho=1.01)
```

Read it top-down:
1. `overloads=[...]` empty and `max_rho` < 0.9 → grid calm, pick a move.
2. `overloads` non-empty → that line trips on its 3rd consecutive overloaded
   step (above: hot since t=1, so it trips at t=3 unless relieved by your
   next act); make relieving it YOUR next act.
3. `lines_down` rose → something is open. Find it in `observe --detailed`
   (`status: "down"`; `cooldown > 0` means it tripped and can't be closed yet).
4. Last line → what your previous act did, `no trips` or `tripped: [...]`.
   It is absent before your first act.

Full detail (every line's rho/status, subs, gens — the C3 JSON):

```
simctl observe --detailed
```
```json
{"t": 2, "max_t": 24, "max_rho": 1.0125, "n_down": 1, "n_overflow": 1,
 "lines": [{"id": 1, "name": "0_4_1", "or": "sub_0", "ex": "sub_4", "rho": 0.0,
            "p_or": 0.0, "p_ex": 0.0, "status": "down", "overflow": false,
            "cooldown": 0, "maint": -1},
           {"id": 4, "name": "1_4_4", "or": "sub_1", "ex": "sub_4", "rho": 1.0125,
            "p_or": 43.605, "p_ex": -42.583, "status": "up", "overflow": true,
            "cooldown": 0, "maint": -1}, ...],
 "gens": [{"id": 0, "name": "gen_1_0", "sub": "sub_1", "p": 73.9,
           "renewable": false, "redispatchable": true}, ...],
 "last_action": {"source": "agent", "summary": "set_line_status 0_4_1=down",
                 "args": {"set_line_status": {"0_4_1": -1}}, "t": 0},
 "last_disc_lines": [], "cause": null, ...}
```
(`...` marks elided entries.) Machine-readable version of any command:
append `--json`.
