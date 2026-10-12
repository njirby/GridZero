# Recipe: check the state

**When:** start of your session, after any act, whenever unsure. This is the
read-only pair — it never changes the grid. (Outputs below are from an
`l2rpn_case14_sandbox` run; names and numbers differ on other grids.)

```
simctl status
```
```
sim up · env=l2rpn_case14_sandbox · t=0 · reward=-10.0 (cum 0.0) · done=no
```
`t=0` = step 0; each act/step advances it by one. The `-10.0` at t=0 is a placeholder before any step. `done=no` = running.
`sim down` or a backend error (exit 2) → retry `simctl status`; you cannot
reset the episode — if the sim stays down, the run is over. After
`done=yes` (cause `time_exceeded` or `game_over`), stop and write your
summary.

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
3. `lines_down` rose → something is open. Find it with
   `simctl list lines --where status=down` (a `cooldown=N` flag means it
   tripped and can't be closed yet).
4. Last line → what your previous act did, `no trips` or `tripped: [...]`.
   It is absent before your first act; a rejected act ends with
   `(rejected, not applied)`.

Full detail — every line, sub and gen, one row each (`observe --detailed`
prints the same):

```
simctl grid
```
```
t=1 reward=63.4 (cum 63.4) done=no · max_rho=1.019 n_down=1 n_overflow=1 overflow_allowed=2 · sim_clock=2019-01-06 00:05:00
since your last act (t=0 set_line_status 0_4_1=down): no trips
LINES (20) — name rho p_or(MW) [flags]; flags only when not default: status (down/cooldown/maintenance), cooldown=N, maint=now|in N, or_bus=/ex_bus= (default 1; -1 when down), overflow k/allowed
  0_1_0     0.579     73.0
  0_4_1     0.000      0.0 down
  ...
  1_4_4     1.019     43.9 overflow 1/2
  ...
SUBS (14) — name type net_p(MW, gen positive)
  sub_0  gen       73.0
  ...
GENS (6) — name sub p(MW) R=redispatchable W=renewable [flags: disconnected, dispatch target= actual= (MW, when non-zero)]
  gen_1_0 sub_1      73.3 R-
  ...
(every field of one element: simctl get <line|sub|gen>.<name>; all fields as JSON: simctl --json grid)
```
(`...` marks elided rows.) Row formats and flags: `docs/grid2op/observation.md`.
Machine-readable version of any command: append `--json`.

Narrower reads:

```
simctl get sub.sub_4                     # every field + attached lines (end, busbar) and gens
simctl get line.1_4_4                    # every field of one line
simctl list lines --sort rho --top 5     # the 5 hottest lines
simctl list lines --where 'cooldown>0'   # lines that can't be switched yet
simctl history line.1_4_4 --last 6       # one line's rho/p_or/status per step
```
```
$ simctl list lines --sort rho --top 5
5 of 20 matching lines (of 20) · sorted by rho descending, top 5 — name rho p_or(MW) [flags]; …
  5_12_9    0.799     20.6
  5_10_7    0.667     14.2
  1_4_4     0.643     27.7
  5_11_8    0.522      8.3
  11_12_13  0.509      2.7
```
(at t=0; header shortened with `…`.)

On a big grid, read `simctl grid` once at the start, then each turn
`simctl observe` plus:

```
simctl observe --changes
```
It prints the `grid` rows only for the lines/subs/gens that changed since
your last `grid` or `--changes` view. Discrete changes (status, bus,
overflow, cooldown start/end) always show; rho/MW drift only above the
thresholds printed in its header (`--rho-eps`, `--mw-eps` to change them).
