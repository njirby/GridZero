# Recipe: see the grid (render + Read) — OPTIONAL

**When:** you need TOPOLOGY visually — which lines connect which subs, where
the generators sit. This is optional: `observe --detailed` already gives the
topology as numbers (each line's `or`/`ex` subs, each sub's `x`/`y`), and in
some environments (e.g. RL training) render is turned off.

## Step 1 — render

```
simctl render
```
```
wrote /home/nate/Documents/GridZero/render/18751/t0000.png (800x500)
```
It prints the absolute path of the PNG it just wrote (the directory depends
on the deployment; use the path printed). Optional name:
`simctl render --out mygrid.png` — `--out` takes a **bare filename only**
(`[A-Za-z0-9_.-]`, ending `.png`); anything with a `/` or `..` is refused:

```
invalid render filename: use a bare name like 't0052' or 't0052.png'
```

**If render is disabled** the output is just:

```
render is disabled in this environment
```
(exit 1). Don't retry; use `observe --detailed`.

## Step 2 — Read it

Use your `Read` tool on the printed path.

What you'll see: 14 substations as labeled nodes on a roughly ring-shaped
grid, 20 lines as segments between them colored by loading, open lines
drawn differently, and generator icons on the gen subs.

## Reading the layout

- A line labeled `a_b_c` runs from `sub_a` to `sub_b`.
- Subs that share lines are coupled — e.g. sub_4 has four feeders (0_4_1,
  1_4_4, 3_4_6, 4_5_17): opening one re-routes through the others.
- Gen subs (icons) are your `redispatch` levers; a hot line coming OFF a gen
  sub → pull that gen down.
- `observe --detailed` has the same geometry as numbers: each line's `or`/`ex`
  fields name its two subs, and each sub's `x`/`y` are the render coordinates.

## Then act

The render is for deciding. Once you know the topology, go back to the loop
(`act` + `observe`) — the PNG does not update itself each step.
