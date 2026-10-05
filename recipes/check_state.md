# Recipe: check the state

**When:** start of your session, after any act, whenever unsure. This is the
read-only pair — it never changes the grid.

```
simctl status
```
```
sim up · env=l2rpn_case14_sandbox · t=50/8064 · reward=63.1 (cum 3120.4) · done=no
```
`done=no` = episode running. `sim down` or a backend error (exit 2) → run
`simctl reset`.

```
simctl observe
```
```
t=52 reward=61.2 (cum 3074.8) done=no
lines_down=1  max_rho=0.97 (2_3_5)  overloads=[2_3_5]
top loads:  2_3_5=97.0%  0_4_1=88.0%  5_12_9=71.0%
gens: 81.4 79.3 5.3 0.0 | loads: 5.4 12.6 14.4 ...
since your last act (t=51 set_line_status 0_4_1=-1): 2_3_5 +9.0%, no trips
```

Read it top-down:
1. `overloads=[...]` empty and `max_rho` < 0.9 → grid calm, pick a move.
2. `overloads` non-empty → that line trips next step if not relieved; make
   relieving it YOUR next act.
3. Last line → what your previous act did; a trip named here means you
   caused it.

Full detail (every line's rho/status, subs, gens — the C3 JSON):

```
simctl observe --detailed
```
```json
{"t": 50, "max_t": 8064, "max_rho": 1.1737, "n_down": 1, "n_overflow": 1,
 "lines": [{"name": "0_4_1", "or": "sub_0", "ex": "sub_4", "rho": 0.0,
            "status": "down", "overflow": false, "cooldown": 0, "maint": -1},
           {"name": "1_4_4", "or": "sub_1", "ex": "sub_4", "rho": 1.1737,
            "status": "up", "overflow": true, "cooldown": 0, "maint": -1}],
 ...}
```

Machine-readable version of any of the above: append `--json`.
