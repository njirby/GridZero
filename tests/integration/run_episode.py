#!/usr/bin/env python
"""E — integration/eval driver.

Runs ONE real episode end-to-end: boots the backend (which owns the grid2op env
AND drives opencode/AA-Dense-Blackwell), lets the model operate the grid for a
time budget, records every C4 event, and reports whether the v0 acceptance gate
passed (model produced >=1 legal act, transcript shows reasoning + tool +
outcome, and the grid state advanced).

This is the Phase-2 acceptance test. It requires WS A (backend), WS B (simctl
on PATH in the opencode env), and WS C (docs/AGENTS.md) to be present.

Run:
  cd /home/nate/Documents/GridZero
  ./.venv/bin/python tests/integration/run_episode.py [--budget 180] [--port 8731]

Outputs:
  runs/eval-<ts>.jsonl   (every C4 event)
  runs/eval-<ts>.report.json  (scorecard)
"""
import argparse, json, os, subprocess, sys, time, threading
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "runs")
os.makedirs(RUNS, exist_ok=True)

import httpx  # in .venv


def wait_up(base, path, timeout=150):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(base + path, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_backend(port):
    log = open(os.path.join(RUNS, f"backend-{port}.log"), "ab")
    import secrets
    tok = secrets.token_hex(16)
    env = dict(os.environ, SIM_API_TOKEN=tok)
    p = subprocess.Popen(
        [os.path.join(ROOT, ".venv/bin/python"), "-m", "uvicorn",
         "backend.app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    return p, tok


def collect_events(base, out, stop, headers=None):
    n = {"sim_state": 0, "step_outcome": 0, "agent_delta": 0,
         "agent_reasoning": 0, "tool_call": 0, "tool_result": 0, "turn_end": 0,
         "agent_steps": 0, "legal_agent_acts": 0, "t_start": None, "t_end": None,
         "cum_reward_end": None}
    with out, httpx.stream("GET", base + "/event", headers=headers, timeout=None) as resp:
        for line in resp.iter_lines():
            if stop():
                break
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except Exception:
                continue
            try:
                out.write(json.dumps(ev) + "\n")
            except Exception:
                pass
            t = ev.get("type")
            d = ev.get("data", {})
            if t == "sim.state":
                n["sim_state"] += 1
                if n["t_start"] is None:
                    n["t_start"] = d.get("t")
                n["t_end"] = d.get("t")
                n["cum_reward_end"] = d.get("cum_reward")
            elif t == "sim.step_outcome":
                n["step_outcome"] += 1
                if d.get("source") == "agent":
                    n["agent_steps"] += 1
                    if not d.get("illegal") and not d.get("ambiguous"):
                        n["legal_agent_acts"] += 1
            elif t == "agent.delta":
                n["agent_delta"] += 1
                if d.get("field") == "reasoning":
                    n["agent_reasoning"] += 1
            elif t == "agent.tool_call":
                n["tool_call"] += 1
            elif t == "agent.tool_result":
                n["tool_result"] += 1
            elif t == "agent.turn_end":
                n["turn_end"] += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--budget", type=float, default=180.0, help="seconds to let the model run")
    ap.add_argument("--no-backend", action="store_true", help="assume backend already running")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    ts = time.strftime("%Y%m%d-%H%M%S")
    evfile = os.path.join(RUNS, f"eval-{ts}.jsonl")
    reportf = os.path.join(RUNS, f"eval-{ts}.report.json")

    bp = None
    tok = ""
    if not a.no_backend:
        bp, tok = start_backend(a.port)
        print(f"backend pid {bp.pid}; waiting for {base}/sim/status (cold boot can take ~60-90s) ...")
        if not wait_up(base, "/sim/status"):
            print("ERROR: backend did not come up; see", os.path.join(RUNS, f"backend-{a.port}.log"))
            bp.terminate()
            try:
                bp.wait(timeout=10)
            except Exception:
                bp.kill()
            return 2
    final_state = {}
    try:
        # fresh episode: reset sim + (re)start opencode session + kickoff
        hdr = {"Authorization": "Bearer " + tok} if tok else {}
        httpx.post(base + "/sim/reset", json={}, headers=hdr, timeout=60)
        r = httpx.post(base + "/control", json={"cmd": "reset", "args": {}}, headers=hdr, timeout=60)
        print("control reset:", r.status_code, r.text[:200])
        time.sleep(2)
        deadline = time.time() + a.budget
        stop = lambda: time.time() > deadline
        with open(evfile, "w") as out:
            n = collect_events(base, out, stop, hdr)
        # capture final state BEFORE the backend is torn down in `finally`
        try:
            final_state = httpx.get(base + "/sim/state", headers=hdr, timeout=10).json().get("data", {})
        except Exception as e:
            print("final_state fetch failed:", e)
    finally:
        if bp:
            bp.terminate()
            try:
                bp.wait(timeout=10)
            except Exception:
                bp.kill()

    passed = (n["legal_agent_acts"] >= 1
              and n["agent_reasoning"] >= 1
              and n["tool_result"] >= 1
              and (n["t_end"] or 0) > (n["t_start"] or 0))
    report = {"ts": ts, "budget_s": a.budget, "counts": n,
              "final": {"t": final_state.get("t"), "cum_reward": final_state.get("cum_reward"),
                        "max_rho": final_state.get("max_rho"), "n_down": final_state.get("n_down")},
              "events_file": evfile, "PASS": passed,
              "gate": ">=1 legal agent act + reasoning + tool result + t advanced"}
    json.dump(report, open(reportf, "w"), indent=2)
    print("\n=== EVAL SCORECARD ===")
    print(json.dumps(report, indent=2))
    print("PASS" if passed else "FAIL", "(v0 acceptance gate)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
