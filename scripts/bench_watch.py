#!/usr/bin/env python
"""scripts/bench_watch.py — general watcher for a benchmark LLM run.

Polls a set of backend ports for live progress + checks for landed results.json
files (episode finished) matching a horizon. Appends to a watch log; prints a
JSON summary on the last line. Used by watcher subagents.

Usage:
  python scripts/bench_watch.py --ports 8880,8881,8882 --horizon 96 \
      --chronics 0,1,100,500,750,900 --log /tmp/opencode/pilot_watch.log
"""
import argparse, json, os, sys, glob, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def landed_for(horizon, chronics):
    """{chronic: results_path} for finished episodes at this horizon/chronic set.

    Prefers the NEWEST outdir per chronic (dir name encodes a start timestamp) so a
    stale/broken episode can never shadow a fresh one. Skips agent_failed episodes
    (infra failures) — they are not a valid 'landed' result.
    """
    best = {}  # chronic -> (mtime, rf)
    for rf in glob.glob(os.path.join(ROOT, "runs", "llm-*", "results.json")):
        try:
            res = json.load(open(rf))
        except Exception:
            continue
        for r in res:
            if r.get("horizon") != horizon or r.get("chronic") not in chronics:
                continue
            if r.get("agent_failed"):
                continue
            mtime = os.path.getmtime(rf)
            c = r["chronic"]
            if c not in best or mtime > best[c][0]:
                best[c] = (mtime, rf)
    return {c: rf for c, (_, rf) in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--chronics", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    ports = [int(p) for p in a.ports.split(",") if p]
    chronics = {int(c) for c in a.chronics.split(",") if c}

    landed = landed_for(a.horizon, chronics)
    running = {}
    try:
        import httpx
        for p in ports:
            try:
                s = httpx.get(f"http://127.0.0.1:{p}/bench/stats", timeout=3).json()["sim"]
                running[p] = {"chronic": s.get("chronic"), "t": s.get("t"),
                              "trips": s.get("n_trips"), "cum": round(s.get("cum_reward", 0))}
            except Exception:
                running[p] = None
    except Exception:
        pass
    active = {p: v for p, v in running.items() if v}
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (f"[{ts}] LANDED {sorted(landed)} ({len(landed)}/{len(chronics)}) | "
            f"RUNNING { {p: str(v['chronic'])+'@t'+str(v['t']) for p, v in active.items()} or 'none'}")
    print(line)
    with open(a.log, "a") as f:
        f.write(line + "\n")
    summary = {"landed": sorted(landed), "n_landed": len(landed), "n_total": len(chronics),
               "landed_paths": landed, "running": {str(p): v for p, v in active.items()},
               "n_running": len(active)}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
