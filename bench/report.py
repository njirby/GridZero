#!/usr/bin/env python
"""bench/report.py — aggregate benchmark results into a leaderboard.

Ingests:
  --baselines <jsonl>   from run_baselines.py (DoNothing rows are the anchors)
  --llm <json> [...]    from run_llm.py results.json (lists of EpisodeResult)
Normalizes every agent against the DoNothing anchor for the SAME (chronic,
horizon), then reports, per agent:
  headline  = mean improvement-over-DN  (DN pinned to 0)  + bootstrap 95% CI
  norm      = mean L2RPN-scale normalized score           + CI
  survival% = fraction of episodes reaching the horizon
  safety    = mean trips, game-overs, peak-rho
  cost      = mean $, wall-clock, tokens, LLM turns (LLM agents only)
  efficiency= improvement per $ (the "economical operator" axis)
Plus a per-chronic heatmap and a rule-based error taxonomy.

Output: <out>/leaderboard.md, <out>/leaderboard.html, <out>/summary.json
"""
import argparse, glob, json, os, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from bench.score import EpisodeResult, normalize


def load_episode_list(path):
    if path.endswith(".jsonl"):
        return [EpisodeResult(**_strip(r)) for r in (json.loads(l) for l in open(path) if l.strip())]
    data = json.load(open(path))
    if isinstance(data, dict) and "agent" in data:
        data = [data]
    return [EpisodeResult(**_strip(r)) for r in data]


def _strip(r):
    keep = {k: r.get(k) for k in EpisodeResult.__dataclass_fields__}
    return keep


def bootstrap_ci(values, n=2000, alpha=0.05, seed=0):
    values = [v for v in values if v is not None]
    if not values:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (float(vals.mean()), lo, hi)


def error_taxonomy(res, dn):
    """Classify one episode: outcome-vs-DN + (for game-overs) the death mechanism.

    Actionable buckets (from the real @1200 runs):
      OUTLASTED-DN            — survived; DN died here  (a WIN)
      completed               — both survived; reward decides
      tied-DN-death           — died within 8 steps of DN
      died-earlier-than-DN    — died before DN  (worse than passive)
      outlasted-DN-partial    — died but AFTER DN
      DIED-WHEN-DN-SURVIVED   — DN finished, agent game-overed  (clearly worse)
      instant-collapse        — died in the first 10 steps
      repeated-illegal        — mostly rejected (hallucinated) actions
    Death mechanism: loss-of-load/non-trip vs trip-cascade (N trips).
    """
    if res.done:
        if dn.game_over:
            return f"OUTLASTED-DN (survived; DN died @ {dn.survived})"
        return "completed (DN also survived — reward decides)"
    mechanism = ("loss-of-load/non-trip" if res.n_trips == 0
                 else f"trip-cascade ({res.n_trips} trips)")
    if res.survived <= 10:
        return f"instant-collapse ({mechanism})"
    if res.n_illegal > max(1, res.survived) * 0.2:
        return f"repeated-illegal ({res.n_illegal} rejected; {mechanism})"
    if dn.game_over:
        diff = res.survived - dn.survived
        if abs(diff) <= 8:
            tag = f"tied-DN-death (~{res.survived}; DN {dn.survived})"
        elif diff < 0:
            tag = f"died-earlier-than-DN ({res.survived}<{dn.survived})"
        else:
            tag = f"outlasted-DN-partial ({res.survived}>{dn.survived})"
        return f"{tag} | {mechanism}"
    return f"DIED-WHEN-DN-SURVIVED ({res.survived}; DN {dn.survived}) | {mechanism}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="append", default=[])
    ap.add_argument("--llm", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="l2rpn_case14_sandbox")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    allres = {}  # agent -> [EpisodeResult]
    infra_failures = []  # agent_failed episodes (excluded from scoring)
    for path in a.baselines + a.llm:
        for r in load_episode_list(path):
            if getattr(r, "agent_failed", False):
                infra_failures.append(r)
                continue
            allres.setdefault(r.agent, []).append(r)

    # DN anchors: (chronic, horizon) -> DN result (mean cum if repeats)
    anchors = {}
    for r in allres.get("DoNothing", []):
        key = (r.chronic, r.horizon)
        anchors.setdefault(key, []).append(r)
    anchors = {k: v[0] for k, v in anchors.items()}  # one DN per (chronic,horizon)

    # score every agent against its anchor
    rows = []
    for agent, res_list in allres.items():
        for res in res_list:
            dn = anchors.get((res.chronic, res.horizon))
            if dn is None:
                continue
            sc = normalize(res, dn)
            rows.append({"agent": agent, **res.to_dict(), **sc,
                         "error_bucket": error_taxonomy(res, dn)})

    # aggregate per agent
    agents = {}
    for row in rows:
        agents.setdefault(row["agent"], []).append(row)

    summary = {"env": a.env, "n_anchors": len(anchors), "agents": {}}
    base_order = ["DoNothing", "Random", "RecoPowerline", "N1Greedy"]
    order = base_order + [ag for ag in agents if ag not in base_order]
    for agent in order:
        if agent not in agents:
            continue
        rs = agents[agent]
        impr = [r["improvement"] for r in rs]
        normv = [r["norm"] for r in rs]
        surv = [r["done"] for r in rs]
        cost = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
        wall = [r["wall_clock_s"] for r in rs if r["wall_clock_s"] is not None]
        turns = [r["llm_turns"] for r in rs if r["llm_turns"] is not None]
        tok_in = [r["tokens_in"] for r in rs if r["tokens_in"] is not None]
        agg = {
            "n_episodes": len(rs),
            "improvement_mean": bootstrap_ci(impr)[0],
            "improvement_ci95": bootstrap_ci(impr)[1:],
            "norm_mean": bootstrap_ci(normv)[0],
            "norm_ci95": bootstrap_ci(normv)[1:],
            "survival_pct": 100 * statistics.mean(surv) if surv else None,
            "game_overs": sum(1 for r in rs if r["game_over"]),
            "trips_mean": statistics.mean([r["n_trips"] for r in rs]),
            "peak_rho_mean": statistics.mean([r["peak_rho"] for r in rs]),
            "illegal_mean": statistics.mean([r["n_illegal"] for r in rs]),
            "cost_mean": statistics.mean(cost) if cost else None,
            "wall_mean": statistics.mean(wall) if wall else None,
            "turns_mean": statistics.mean(turns) if turns else None,
            "tokens_in_mean": statistics.mean(tok_in) if tok_in else None,
            "efficiency_impr_per_usd": (bootstrap_ci(impr)[0] / statistics.mean(cost)) if cost and statistics.mean(cost) else None,
        }
        summary["agents"][agent] = agg

    summary["infra_failures"] = [
        {"agent": r.agent, "chronic": r.chronic, "horizon": r.horizon,
         "survived": r.survived, "notes": r.notes, "ep": r.ep} for r in infra_failures]
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    write_markdown(a.out, summary, agents, order)
    write_html(a.out, summary, agents, order)
    print(f"wrote {a.out}/leaderboard.md + .html + summary.json")
    print_leaderboard(summary, order)
    if infra_failures:
        print(f"\nINFRA FAILURES (excluded from scoring): {len(infra_failures)}")
        for r in infra_failures:
            print(f"  {r.agent} chronic={r.chronic}@{r.horizon} survived={r.survived} — {r.notes}")


def fmt(x, spec=".3f"):
    return "—" if x is None else format(x, spec)


def print_leaderboard(summary, order):
    print(f"\n=== {summary['env']} leaderboard (improvement over do-nothing, DN=0) ===")
    print(f"{'agent':18s} {'impr(mean±ci95)':>20s} {'norm(mean)':>11s} {'surv%':>6s} "
          f"{'games':>5s} {'trips':>6s} {'$':>7s} {'wall(s)':>8s} {'n':>3s}")
    for agent in order:
        if agent not in summary["agents"]:
            continue
        g = summary["agents"][agent]
        lo, hi = g["improvement_ci95"]
        impr = f"{fmt(g['improvement_mean'],'.4f')} [{fmt(lo,'.4f')},{fmt(hi,'.4f')}]"
        print(f"{agent:18s} {impr:>20s} {fmt(g['norm_mean'],'.1f'):>11s} "
              f"{fmt(g['survival_pct'],'.0f'):>6s} {g['game_overs']:>5d} "
              f"{fmt(g['trips_mean'],'.2f'):>6s} {fmt(g['cost_mean'],'.3f'):>7s} "
              f"{fmt(g['wall_mean'],'.0f'):>8s} {g['n_episodes']:>3d}")


def write_markdown(out, summary, agents, order):
    L = [f"# Grid2op LLM Benchmark — {summary['env']}", ""]
    L.append("Headline metric: **improvement over do-nothing** (DoNothing pinned to 0). "
             "`norm` is the L2RPN-style [−100,0,80,100] scale. Bootstrap 95% CI over episodes.")
    L.append("")
    L.append("| agent | improvement (mean) | CI95 | norm (mean) | survival % | game-overs | trips | peak-ρ | $/ep | wall(s) | LLM turns | n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for agent in order:
        if agent not in summary["agents"]:
            continue
        g = summary["agents"][agent]
        lo, hi = g["improvement_ci95"]
        L.append(f"| {agent} | {fmt(g['improvement_mean'],'.4f')} | [{fmt(lo,'.4f')}, {fmt(hi,'.4f')}] | "
                 f"{fmt(g['norm_mean'],'.1f')} | {fmt(g['survival_pct'],'.0f')} | {g['game_overs']} | "
                 f"{fmt(g['trips_mean'],'.2f')} | {fmt(g['peak_rho_mean'],'.3f')} | {fmt(g['cost_mean'],'.3f')} | "
                 f"{fmt(g['wall_mean'],'.0f')} | {fmt(g['turns_mean'],'.0f')} | {g['n_episodes']} |")
    L.append("")
    L.append("## Efficiency (improvement per $)")
    L.append("| agent | improvement/$ |")
    L.append("|---|---|")
    for agent in order:
        if agent in summary["agents"]:
            g = summary["agents"][agent]
            L.append(f"| {agent} | {fmt(g['efficiency_impr_per_usd'],'.4f')} |")
    L.append("")
    L.append("## Per-episode detail")
    for agent in order:
        if agent not in agents:
            continue
        L.append(f"### {agent}")
        L.append("| chronic | horizon | survived | cum | improvement | norm | trips | illegal | $ | error bucket |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(agents[agent], key=lambda x: (x["horizon"], x["chronic"])):
            L.append(f"| {r['chronic']} | {r['horizon']} | {r['survived']} | {fmt(r['cum_reward'],'.0f')} | "
                     f"{fmt(r['improvement'],'.4f')} | {fmt(r['norm'],'.1f')} | {r['n_trips']} | "
                     f"{r['n_illegal']} | {fmt(r['cost_usd'],'.3f')} | {r['error_bucket']} |")
        L.append("")
    open(os.path.join(out, "leaderboard.md"), "w").write("\n".join(L))


def write_html(out, summary, agents, order):
    # simple self-contained HTML leaderboard
    rows = []
    for agent in order:
        if agent not in summary["agents"]:
            continue
        g = summary["agents"][agent]
        lo, hi = g["improvement_ci95"]
        cls = "dn" if agent == "DoNothing" else ("bad" if (g["improvement_mean"] or 0) < -0.1 else "good")
        rows.append(
            f"<tr class='{cls}'><td>{agent}</td><td>{fmt(g['improvement_mean'],'.4f')}</td>"
            f"<td>{fmt(lo,'.4f')} … {fmt(hi,'.4f')}</td><td>{fmt(g['norm_mean'],'.1f')}</td>"
            f"<td>{fmt(g['survival_pct'],'.0f')}%</td><td>{g['game_overs']}</td>"
            f"<td>{fmt(g['trips_mean'],'.2f')}</td><td>{fmt(g['peak_rho_mean'],'.3f')}</td>"
            f"<td>{fmt(g['cost_mean'],'.3f')}</td><td>{fmt(g['wall_mean'],'.0f')}</td>"
            f"<td>{fmt(g['turns_mean'],'.0f')}</td><td>{g['n_episodes']}</td></tr>")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Grid2op LLM Benchmark</title>
<style>body{{font-family:ui-monospace,Menlo,monospace;background:#0a0e14;color:#d7e0ea;margin:24px}}
h1{{color:#38bdf8}}table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 10px;border-bottom:1px solid #1e2836;text-align:right}}
td:first-child,th:first-child{{text-align:left}}tr.dn td{{color:#6b7a8c}}tr.good td{{color:#22c55e}}tr.bad td{{color:#ef4444}}
.note{{color:#6b7a8c;font-size:12px}}</style></head><body>
<h1>Grid2op LLM Benchmark — {summary['env']}</h1>
<p class='note'>Headline: improvement over do-nothing (DN=0). CI95 = bootstrap over episodes. norm = L2RPN [−100,0,80,100] scale.</p>
<table><tr><th>agent</th><th>improvement</th><th>CI95</th><th>norm</th><th>survival</th><th>game-overs</th>
<th>trips</th><th>peak-ρ</th><th>$ /ep</th><th>wall(s)</th><th>LLM turns</th><th>n</th></tr>
{''.join(rows)}</table></body></html>"""
    open(os.path.join(out, "leaderboard.html"), "w").write(html)


if __name__ == "__main__":
    main()
