# Recipe: see the grid (render + Read)

**When:** you need TOPOLOGY — which lines connect which subs, where the
generators sit, how the ring is wired. Text `observe` tells you loadings;
the PNG tells you shape. Don't render every turn (~750 ms + ~676 tokens to
look); render when a topology question is blocking a decision.

## Step 1 — render

```
simctl render
```
```
wrote /home/nate/grid2op-harness/render/t0052.png (800x500)
```
It prints the absolute path of the PNG it just wrote. Options:
`simctl render --out /tmp/mygrid.png --width 1200`.

## Step 2 — Read it

Use your `Read` tool on the printed path:

```
Read /home/nate/grid2op-harness/render/t0052.png
```

What you'll see: 14 substations as labeled nodes on a roughly ring-shaped
grid (sub_0 bottom-left, sub_4 center, sub_13 right, …), 20 lines as
segments between them, colored by loading (red = hot, green = light), broken
segments = open lines, and generator icons on sub_0/sub_1/sub_2/sub_5/sub_7.

## Reading the layout

- A line labeled `a_b_c` runs from `sub_a` to `sub_b`.
- Subs that share two lines are coupled — e.g. sub_1 and sub_4 are connected
  by `1_4_4` AND both feed sub_4's ring via `0_4_1` (sub_0→sub_4) and
  `3_4_6` (sub_3→sub_4): opening one of those re-routes through the others.
  sub_4 has four feeders total (0_4_1, 1_4_4, 3_4_6, 4_5_17).
- Gen subs (icons) are your `redispatch` levers; a hot line coming OFF a gen
  sub → pull that gen down.
- `observe --detailed` has the same geometry as numbers: each line's `or`/`ex`
  fields name its two subs, and each sub's `x`/`y` are the render coordinates.
  Cross-reference when the PNG is ambiguous.

## Then act

The render is for deciding. Once you know the topology, go back to the loop
(`act` + `observe`) — the PNG does not update itself each step. Re-render
only after a structural change you want to re-verify visually.
