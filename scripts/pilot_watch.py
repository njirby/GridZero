#!/usr/bin/env python
"""scripts/pilot_watch.py — watcher helper for the pilot run.

Called repeatedly by the watcher subagent. Reports, per poll:
  - which pilot episodes have LANDED (a completed runs/llm-<chronic>-*/results.json
    with horizon=96 — run_llm.py only writes this when an episode reaches
    done/game_over/safety-cap, so its presence == a finished episode), and
  - which backends are currently RUNNING (live /bench/stats t/trips/cum).
Appends a timestamped line to the watch log and prints a machine-readable JSON
summary (last line of stdout) so the subagent can make the stop decision.

Exit code is always 0 (the subagent decides when to stop from the summary).
"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LOG = os.environ.get("PILOT_WATCH_LOG", "/tmp/opencode/pilot_watch.log")
PILOT_CHRONICS = {0, 1, 100, 500, 750, 900}
HORIZON = 96
PORTS = range(8820, 8826)


def landed_episodes():
    """{chronic: results_path} for pilot episodes that finished (results.json exists)."""
    landed = {}
    runs = os.path.join(ROOT, "runs")
    for d in os.listdir(runs):
        if not d.startswith("llm-"):
            continue
        rf = os.path.join(runs, d, "results.json")
        if not os.path.isfile(rf):
            continue
        try:
            res = json.load(open(rf))
        except Exception:
            continue
        for r in res:
            if r.get("horizon") == HORIZON and r.get("chronic") in PILOT_CHRONICS:
                landed[r["chronic"]] = rf
    return landed


def snapshot():
    out = {}
    try:
        import httpx
    except Exception:
        return out
    for port in PORTS:
        try:
            s = httpx.get(f"http://127.0.0.1:{port}/bench/stats", timeout=3).json()["sim"]
            out[port] = {"chronic": s.get("chronic"), "t": s.get("t"), "trips": s.get("n_trips"),
                         "cum": round(s.get("cum_reward", 0)), "done": s.get("done"),
                         "game_over": s.get("game_over")}
        except Exception:
            out[port] = None
    return out


def main():
    landed = landed_episodes()
    snap = snapshot()
    running = {p: v for p, v in snap.items() if v}
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    run_desc = {p: f"{v['chronic']}@t{v['t']}(trips {v['trips']})" for p, v in running.items()}
    line = (f"[{ts}] LANDED {sorted(landed)} ({len(landed)}/6) | "
            f"RUNNING {run_desc if run_desc else 'none'}")
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    summary = {"landed": sorted(landed.keys()), "n_landed": len(landed),
               "landed_paths": {k: v for k, v in landed.items()},
               "running": {str(p): v for p, v in running.items()},
               "n_running": len(running)}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
