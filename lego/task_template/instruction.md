You are operating the live grid2op simulation `$env` (scenario chronic
#$chronic, seed $seed). `AGENTS.md` is loaded automatically: it is your
operator manual, including the Grid facts for this grid. Map rendering is
disabled here.

**Task:** keep the grid stable (no line overloads or protection trips) and
maximize your cumulative reward until `simctl status` shows `done=yes`. You
are judged on how long the grid survives and how well you keep reward up,
compared with simply doing nothing: a blackout ends the episode early and
forfeits every remaining step.
