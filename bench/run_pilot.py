#!/usr/bin/env python
"""bench/run_pilot.py — run the PILOT panel: the LLM (and optionally the baselines)
across all pilot chronics, in parallel (bounded concurrency).

Each chronic gets its own backend (own port) + own opencode session (own port), so
episodes are isolated and can run concurrently. Thinking time is unlimited (skill,
not speed); a per-episode safety cap checkpoints rather than corrupts.

Usage:
  ./.venv/bin/python bench/run_pilot.py                 # LLM only, 6 pilot chronics, concurrency 2
  ./.venv/bin/python bench/run_pilot.py --conc 3 --horizon 96
  ./.venv/bin/python bench/run_pilot.py --chronics 0,1  # subset
Writes runs/pilot-<ts>/config.json + per-chronic results, then a merged
runs/pilot-<ts>/results.json.
"""
import argparse, concurrent.futures, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = os.path.join(ROOT, ".venv/bin/python")
from bench.panel import load_panel, config_hash


def run_one_chronic(chronic, horizon, port, model, repeats, cap_h, poke_s, config="baseline"):
    """Drive run_llm.py for one chronic on its own port. Returns (chronic, outdir)."""
    cmd = [PY, "bench/run_llm.py", "--chronic", str(chronic), "--horizon", str(horizon),
           "--port", str(port), "--repeats", str(repeats),
           "--safety-cap-h", str(cap_h), "--poke-idle-s", str(poke_s), "--model", model,
           "--config", config, "--liveness-min", "20", "--stall-min", "40"]
    log = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join(log.stdout.splitlines()[-6:])
    print(f"  chronic {chronic}: rc={log.returncode}\n{tail}", flush=True)
    # find the outdir run_llm created (runs/llm-<chronic>-<ts>)
    cands = sorted([d for d in os.listdir(os.path.join(ROOT, "runs"))
                    if d.startswith(f"llm-{chronic}-") and os.path.isfile(
                        os.path.join(ROOT, "runs", d, "results.json"))], reverse=True)
    return chronic, (os.path.join(ROOT, "runs", cands[0]) if cands else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=None)  # default panel pilot
    ap.add_argument("--chronics", default=None, help="comma list; default panel pilot")
    ap.add_argument("--model", default="AA-Dense-Blackwell")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--conc", type=int, default=2, help="parallel episodes")
    ap.add_argument("--base-port", type=int, default=8820)
    ap.add_argument("--safety-cap-h", type=float, default=10)
    ap.add_argument("--poke-idle-s", type=int, default=240)
    ap.add_argument("--config", default="baseline", help="ablation config (bench/configs/<name>.json)")
    a = ap.parse_args()
    panel = load_panel()
    horizon = a.horizon or panel["horizons"]["pilot"]
    chronics = ([int(x) for x in a.chronics.split(",") if x.strip()] if a.chronics
                else panel["pilot_chronics"])
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(ROOT, "runs", f"pilot-{ts}")
    os.makedirs(outdir, exist_ok=True)
    cfg = {"panel": panel, "model": a.model, "horizon": horizon, "chronics": chronics,
           "repeats": a.repeats, "conc": a.conc, "ablation": a.config,
           "config_hash": config_hash(model=a.model, horizon=horizon, panel=panel), "ts": ts}
    json.dump(cfg, open(os.path.join(outdir, "config.json"), "w"), indent=2)
    print(f"pilot: {len(chronics)} chronics x horizon {horizon} x {a.model}, conc={a.conc}")

    merged = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.conc) as ex:
        futs = {ex.submit(run_one_chronic, k, horizon, a.base_port + i, a.model,
                          a.repeats, a.safety_cap_h, a.poke_idle_s, a.config): k
                for i, k in enumerate(chronics)}
        for fut in concurrent.futures.as_completed(futs):
            k, d = fut.result()
            if d:
                merged.append({"chronic": k, "outdir": d,
                               "results": json.load(open(os.path.join(d, "results.json")))})
    json.dump(merged, open(os.path.join(outdir, "results.json"), "w"), indent=2)
    # copy per-chronic results into the pilot dir for the report
    for m in merged:
        for r in m["results"]:
            r["chronic"] = m["chronic"]
            json.dump(r, open(os.path.join(outdir, f"ep-chronic{m['chronic']}-{r.get('ep','x')}.json"), "w"), indent=2)
    print(f"\nwrote {outdir}/results.json ({len(merged)} chronics)")
    print("next: aggregate with bench/report.py --llm " +
          " ".join(f"{m['outdir']}/results.json" for m in merged))


if __name__ == "__main__":
    main()
