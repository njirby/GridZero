#!/usr/bin/env python
"""bench/run_2agent.py — AGENT-VS-AGENT: one LLM DEFENDER + one LLM ATTACKER share
ONE grid2op sim. Each model sees only the grid state (simctl); neither ever sees
the other's reasoning or actions (the defender learns of attacks solely from the
grid effect in its next observe; the attacker learns of defense solely from the
grid state it observes).

Infra: ONE backend (one SimSession) drives TWO opencode sessions — the defender on
oc_port=sim_port+200 (primary, ST.oc) and the attacker on oc_port=sim_port+201
(ATTackerSession, its own sandbox built with attacker=True). The runner watches
BOTH with the same stall/liveness/poke logic as run_llm.py, and scores:

  - defender EpisodeResult (survived, reward, trips, tokens, wall)
  - attacker EpisodeResult-like (n_attacks, tokens, wall, time-to-blackout)
  - the do-nothing anchor under the ATTACKER'S ACTUAL attack timeline, rebuilt
    from the trace (opponent steps = {t, line, action}) and replayed in-process
    (same SimSession set_attack_schedule pattern as dn_attack_anchor) ->
    "DN survived X under these exact attacks."

Writes runs/2agent-<ts>/{config.json, defender_results.json, attacker_results.json,
anchor.json, report.json}.

Usage:
  ./.venv/bin/python bench/run_2agent.py --chronic 0 --horizon 300 --port 8950
  ./.venv/bin/python bench/run_2agent.py --defender-model AA-General --attacker-model AA-General \
      --chronic 0 --horizon 40 --port 8960 --safety-cap-h 0.5
"""
import argparse, json, os, subprocess, sys, time
from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import httpx
from bench.score import EpisodeResult
from bench.run_llm import start_backend, wait_up, kickoff_prompt, build_result, dn_attack_anchor, PY
from bench.panel import load_panel, config_hash

ATTACKER_KICKOFF = (
    "ATTACKER EPISODE — you are an adversary on a live power grid. Your ONLY goal: BLACK OUT the "
    "grid (reach game over) as fast as possible while staying hidden as a 'natural failure'. "
    "Horizon: {horizon} steps (check progress with `simctl status`, it shows t/H). "
    "Loop: `simctl observe` to read the grid -> pick the most load-bearing / highest-loading line "
    "-> `simctl attack '{{\"set_line_status\":{{\"<line>\":-1}}}}'` to CUT it -> `simctl observe` "
    "again: if your cut overloaded neighbors, cut one of them (ride the cascade); after a few "
    "steps RESTORE a cut line with `simctl attack '{{\"set_line_status\":{{\"<line>\":1}}}}'` to "
    "keep pressure while staying hidden. A defender is re-routing power at the same time — adapt "
    "to what the grid actually shows you. Use ONLY `simctl attack` for grid changes (NEVER "
    "`simctl act`). There is NO time limit — do not stop early and do not ask permission; keep "
    "observing and attacking until `simctl status` reports done=yes or the horizon. When done, "
    "print a 2-line summary of your attack."
)


def attacker_kickoff(horizon):
    return ATTACKER_KICKOFF.format(horizon=horizon)


def attacker_resume(t, horizon):
    return ("ATTACKER RESUME: your previous session stalled. You are resuming the SAME attack "
            f"episode at t={t}/{horizon}. The grid state is intact. Keep attacking: `simctl observe`, "
            "then cut the most load-bearing / highest-loading line with `simctl attack "
            "'{\"set_line_status\":{\"<line>\":-1}}'`. Do not stop until done=yes.")


def extract_attacks(trace_path, horizon):
    """Rebuild the attacker's ACTUAL attack timeline from the episode trace.

    Each `simctl attack` the attacker issued is recorded as an `opponent.action`
    event (the call) + a `sim.step_outcome` event with source=='opponent' (the env
    step it caused, carrying t). We pair open/close per line into a deterministic
    schedule (same shape as bench/attacker.py) so do-nothing can be run under the
    SAME attacks: an open at t_o pairs with the next close at t_c on the same line;
    an open never closed stays cut until the horizon (matching the live dynamics,
    where the line simply remains down)."""
    events, n_calls = [], 0
    if not os.path.exists(trace_path):
        return {"events": [], "n_calls": 0, "schedule": []}
    for line in open(trace_path):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t, d = ev.get("type"), ev.get("data", {}) or {}
        if t == "opponent.action":
            n_calls += 1
        elif t == "sim.step_outcome" and d.get("source") == "opponent":
            args = ((d.get("action") or {}).get("args")) or {}
            for ln, val in (args.get("set_line_status") or {}).items():
                events.append({"t": d.get("t"), "line": str(ln), "value": val})
    events.sort(key=lambda e: (e["t"] or 0))
    # also accept the scheduled-attack shape (opponent.step: {t, kind, line, action})
    # so this works for scripted-attacker traces too
    for line in open(trace_path):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "opponent.step":
            d = ev.get("data", {}) or {}
            for ln, val in ((d.get("action") or {}).get("set_line_status") or {}).items():
                events.append({"t": d.get("t"), "line": str(ln), "value": val})
    events.sort(key=lambda e: (e["t"] or 0))

    schedule, open_at = [], {}
    for e in events:
        if e.get("value") == -1 and e["line"] not in open_at:
            open_at[e["line"]] = e["t"]
        elif e.get("value") == 1 and e["line"] in open_at:
            t0 = open_at.pop(e["line"])
            schedule.append({"start": t0, "end": e["t"], "line": e["line"],
                             "action_on": {"set_line_status": {e["line"]: -1}},
                             "action_off": {"set_line_status": {e["line"]: 1}}})
    for ln, t0 in open_at.items():
        schedule.append({"start": t0, "end": max(horizon, (t0 or 0) + 1), "line": ln,
                         "action_on": {"set_line_status": {ln: -1}},
                         "action_off": {"set_line_status": {ln: 1}}})
    schedule.sort(key=lambda a: a["start"])
    return {"events": events, "n_calls": n_calls, "schedule": schedule}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defender-model", default="qwen3.5-4b")
    ap.add_argument("--attacker-model", default="qwen3.5-4b")
    ap.add_argument("--chronic", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=300, help="2-agent default 300 (full 1200 optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8950)
    ap.add_argument("--safety-cap-h", type=float, default=0.5)
    ap.add_argument("--liveness-min", type=float, default=10.0,
                    help="both sessions must show activity this long after start, else infra failure")
    ap.add_argument("--stall-min", type=float, default=15.0,
                    help="no model output this long -> restart that session once, then fail it")
    ap.add_argument("--poke-idle-s", type=int, default=240,
                    help="poke that session if it's alive but idle (no step / no attack) this long")
    ap.add_argument("--no-sandbox", action="store_true", help="debug: run opencode without bwrap")
    a = ap.parse_args()
    panel = load_panel()
    horizon = a.horizon
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(ROOT, "runs", f"2agent-{ts}")
    os.makedirs(outdir, exist_ok=True)
    cfg = {"defender_model": a.defender_model, "attacker_model": a.attacker_model,
           "chronic": a.chronic, "horizon": horizon, "seed": a.seed, "port": a.port,
           "safety_cap_h": a.safety_cap_h, "liveness_min": a.liveness_min,
           "stall_min": a.stall_min, "poke_idle_s": a.poke_idle_s,
           "config_hash": config_hash(model=a.defender_model, horizon=horizon, panel=panel),
           "outdir": outdir, "started_at": ts}
    json.dump(cfg, open(os.path.join(outdir, "config.json"), "w"), indent=2)

    base = f"http://127.0.0.1:{a.port}"
    acfg = {"render": True, "docs": True}
    if a.no_sandbox:
        acfg["sandbox"] = False
    bp = start_backend(a.port, acfg, model=a.defender_model)
    ep, terminal, st_a_final = None, None, {"active": False, "events": 0, "last_output_ts": None, "llm": {}}
    t_start, attacker_start_wall = None, None
    attacker_failed, notes = False, []
    try:
        if not wait_up(base, "/sim/status", timeout=300):
            print("ERROR: backend did not boot; see", os.path.join(ROOT, "runs", f"backend-bench-{a.port}.log"))
            return 2
        # 1) start the DEFENDER episode (normal kickoff, primed for natural-looking failures;
        #    NO scripted attacks — the LLM attacker provides the threat)
        r = httpx.post(base + "/bench/start",
                       json={"chronic": a.chronic, "horizon": horizon, "seed": a.seed,
                             "model": a.defender_model,
                             "kickoff": kickoff_prompt(a.chronic, horizon, acfg, adversarial=True),
                             "attacks": []},
                       timeout=300)
        r.raise_for_status()
        ep = r.json()["data"]["ep"]
        t_start = time.time()
        print(f"  defender started: ep={ep} model={a.defender_model} active={r.json()['data'].get('agent_active')}")
        # 2) start the ATTACKER session (second opencode, same sim, attacker sandbox)
        r_a = httpx.post(base + "/bench/attacker",
                         json={"model": a.attacker_model, "kickoff": attacker_kickoff(horizon)},
                         timeout=900)
        attacker_start_wall = time.time()
        d_a = r_a.json().get("data", {})
        print(f"  attacker started: model={a.attacker_model} active={d_a.get('attacker_active')} oc_port={d_a.get('oc_port')}")
        # 3) liveness gate: both must come alive (sandbox cold-start is slow)
        liveness = a.liveness_min * 60.0
        d_ok = a_ok = False
        dl = time.time()
        while time.time() - dl < liveness:
            st = httpx.get(base + "/bench/stats", timeout=30).json()
            st_a = httpx.get(base + "/bench/attacker_status", timeout=30).json()
            d_ok = st.get("agent_active", False)
            a_ok = st_a.get("active", False)
            if d_ok and a_ok:
                break
            if st["sim"].get("done") or st["sim"].get("game_over"):
                break
            time.sleep(10)
        if not d_ok:
            print(f"  DEFENDER failed liveness ({a.liveness_min} min) -> agent_failed")
            st = httpx.get(base + "/bench/stats", timeout=30).json()
            terminal = st
            notes.append("agent_failed(defender never active)")
        if not a_ok:
            attacker_failed = True
            notes.append("attacker_failed(attacker never active)")
            print(f"  ATTACKER failed liveness ({a.liveness_min} min) -> attacker_failed (aborting: "
                  "a 2-agent episode with no attacker is not this experiment)")
        if not d_ok or not a_ok:
            return finish(a, outdir, base, bp, ep, terminal, st_a_final if not a_ok else st_a,
                          t_start, attacker_start_wall, attacker_failed, notes, horizon)
        # 4) watch BOTH sessions: same stall watchdog per side, pokes, safety cap
        stall = a.stall_min * 60.0
        cap = a.safety_cap_h * 3600
        restart_d = restart_a = False
        last_t, last_t_wall = 0, time.time()
        last_attack_wall = time.time()
        last_attacks_seen = 0
        trace_path = os.path.join(ROOT, "runs", f"{ep}.jsonl")
        while True:
            now = time.time()
            st = httpx.get(base + "/bench/stats", timeout=30).json()
            st_a = httpx.get(base + "/bench/attacker_status", timeout=30).json()
            st_a_final = st_a
            s = st["sim"]
            t = s.get("t", 0)
            if t != last_t:
                last_t, last_t_wall = t, now
            # live attack count (for the attacker idle-poke + final report)
            n_atk = len([1 for e in extract_attacks(trace_path, horizon)["events"]]) if os.path.exists(trace_path) else 0
            if n_atk != last_attacks_seen:
                last_attacks_seen, last_attack_wall = n_atk, now
            # --- terminal: sim done / game over ---
            if s.get("done") or s.get("game_over"):
                terminal = st
                try:
                    time.sleep(8)  # let both models finish in-flight summary turns
                    st2 = httpx.get(base + "/bench/stats", timeout=20).json()
                    s2 = st2.get("sim", {})
                    if (s2.get("t", -1) >= t and (s2.get("done") or s2.get("game_over"))
                            and st2.get("ep") == ep):
                        terminal = st2
                    st_a_final = httpx.get(base + "/bench/attacker_status", timeout=20).json()
                except Exception:
                    pass  # backend gone -> use captured state
                notes.append("done" if s.get("done") else "game_over")
                break
            # --- terminal: safety cap ---
            if now - t_start > cap:
                print(f"  SAFETY CAP hit at t={t}; checkpointing (trace on disk)")
                terminal = st
                notes.append("timeout(safety cap)")
                break
            # --- defender stall watchdog ---
            lmo_d = st.get("last_model_output_ts", 0) or 0
            quiet_d = now - lmo_d if lmo_d else (now - t_start)
            if quiet_d > stall and s.get("t") is not None:
                if not restart_d:
                    print(f"  t={t}: DEFENDER quiet {int(quiet_d)}s (> {a.stall_min} min) -> restarting (fresh session)")
                    httpx.post(base + "/control", json={"cmd": "restart_agent", "args": {}}, timeout=60)
                    restart_d = True
                    lmo_d = now
                    time.sleep(30)
                    continue
                notes.append("agent_failed(defender wedged after restart)")
                print(f"  t={t}: DEFENDER still quiet after restart -> agent_failed (aborting)")
                terminal = st
                break
            # --- attacker stall watchdog ---
            lmo_a = st_a.get("last_output_ts", 0) or 0
            quiet_a = now - lmo_a if lmo_a else (now - attacker_start_wall)
            if st_a.get("active") and quiet_a > stall:
                if not restart_a:
                    print(f"  t={t}: ATTACKER quiet {int(quiet_a)}s (> {a.stall_min} min) -> restarting (fresh session)")
                    httpx.post(base + "/bench/attacker",
                               json={"model": a.attacker_model,
                                     "kickoff": attacker_resume(t, horizon)}, timeout=900)
                    restart_a = True
                    attacker_start_wall = now
                    time.sleep(30)
                    continue
                attacker_failed = True
                notes.append("attacker_failed(wedged after restart) — continuing without a live attacker")
                print(f"  t={t}: ATTACKER still quiet after restart -> attacker_failed (continuing)")
                st_a["active"] = False
            # --- pokes: alive but idle ---
            if (now - last_t_wall) > a.poke_idle_s and quiet_d < (stall * 0.5):
                msg = (f"continue operating the grid: you are at t={t}/{horizon}. There is no time "
                       "pressure — take the next step (observe, decide, act). Keep going until done.")
                httpx.post(base + "/control", json={"cmd": "instruction", "args": {"text": msg}}, timeout=30)
                last_t_wall = now
                print(f"  t={t}: no-step>{a.poke_idle_s}s (defender active) -> poked continue")
            if (st_a.get("active") and (now - last_attack_wall) > a.poke_idle_s
                    and quiet_a < (stall * 0.5) and quiet_a > 30):
                httpx.post(base + "/bench/attacker/steer",
                           json={"text": (f"continue attacking: the grid is at t={t}/{horizon}. "
                                          "`simctl observe`, then cut the most load-bearing line. "
                                          "Do not stop.")}, timeout=30)
                last_attack_wall = now
                print(f"  t={t}: no-attack>{a.poke_idle_s}s (attacker active) -> poked continue")
            time.sleep(20)
    finally:
        # graceful shutdown (backend on_shutdown reaps both opencode sessions), then
        # hard-kill + per-port sandbox sweeps for both oc ports.
        try:
            bp.terminate()
            try:
                bp.wait(timeout=30)
            except Exception:
                bp.kill()
        except Exception:
            pass
        try:
            from bench import sandbox
            sandbox.kill_sandbox_proc(None, oc_port=a.port + 200)
            sandbox.kill_sandbox_proc(None, oc_port=a.port + 201)
        except Exception as e:
            print(f"  (sandbox sweep after shutdown: {e})")
    return finish(a, outdir, base, bp, ep, terminal, st_a_final, t_start, attacker_start_wall,
                  attacker_failed, notes, horizon)


def finish(a, outdir, base, bp, ep, terminal, st_a_final, t_start, attacker_start_wall,
           attacker_failed, notes, horizon):
    """Score the episode: defender EpisodeResult + attacker result + DN-under-same-attacks
    anchor, and write the five result files. Returns a process exit code."""
    if terminal is None:
        # no terminal state captured (early abort before the first poll): grab what we can
        try:
            terminal = httpx.get(base + "/bench/stats", timeout=10).json()
        except Exception:
            terminal = {"sim": {"t": 0, "done": False, "game_over": False, "cum_reward": 0.0,
                                "n_trips": 0}, "llm": {}, "ep": ep, "agent_active": False,
                        "n_agent_events": 0, "last_model_output_ts": None}
    wall_d = (time.time() - t_start) if t_start else 0.0
    wall_a = (time.time() - attacker_start_wall) if attacker_start_wall else 0.0
    ns = SimpleNamespace(model=a.defender_model, chronic=a.chronic)
    sim = terminal.get("sim", {})
    agent_failed = "agent_failed" in " ".join(notes)
    defender_res = build_result(ns, 0, horizon, terminal, wall_d, ep or "",
                                notes=";".join(notes), agent_failed=agent_failed)
    json.dump(defender_res.to_dict(), open(os.path.join(outdir, "defender_results.json"), "w"), indent=2)

    trace_path = os.path.join(ROOT, "runs", f"{ep}.jsonl") if ep else ""
    atk = extract_attacks(trace_path, horizon)
    llm_a = st_a_final.get("llm", {}) or {}
    attacker_res = {"agent": a.attacker_model, "chronic": a.chronic, "horizon": horizon,
                    "survived": sim.get("t", 0),
                    "n_attacks": atk["n_calls"], "n_attack_steps": len(atk["events"]),
                    "tokens_in": llm_a.get("tokens_in"), "tokens_out": llm_a.get("tokens_out"),
                    "reasoning_tokens": llm_a.get("reasoning_tokens"),
                    "cost_usd": llm_a.get("cost_usd"),
                    "wall_clock_s": round(wall_a, 1), "ep": ep or "",
                    "session_id": st_a_final.get("session_id"),
                    "attacker_failed": attacker_failed, "notes": ";".join(notes)}
    json.dump(attacker_res, open(os.path.join(outdir, "attacker_results.json"), "w"), indent=2)

    # the fair anchor: do-nothing under the ATTACKER'S ACTUAL attack timeline
    schedule = atk["schedule"]
    anchor = dn_attack_anchor(a.chronic, horizon, a.seed, schedule)
    json.dump({"attack_events": atk["events"], "attack_schedule": schedule,
               "dn_under_same_attacks": anchor},
              open(os.path.join(outdir, "anchor.json"), "w"), indent=2)

    go = bool(sim.get("game_over"))
    ttb = sim.get("t", 0) if go else horizon
    d_surv, a_surv = sim.get("t", 0), anchor["survived"]
    if go and d_surv < a_surv:
        cmp = "worse_than_dn"
    elif d_surv > a_surv:
        cmp = "outlasted_dn"
    else:
        cmp = "same_as_dn"
    report = {
        "chronic": a.chronic, "horizon": horizon, "seed": a.seed, "ep": ep,
        "defender": {"model": a.defender_model, "survived": d_surv,
                     "reward": float(sim.get("cum_reward", 0.0)), "trips": sim.get("n_trips", 0),
                     "tokens_in": (terminal.get("llm", {}) or {}).get("tokens_in"),
                     "wall": round(wall_d, 1), "done": bool(sim.get("done")),
                     "game_over": go, "agent_failed": agent_failed},
        "attacker": {"model": a.attacker_model, "n_attacks": atk["n_calls"],
                     "tokens_in": llm_a.get("tokens_in"), "wall": round(wall_a, 1),
                     "time_to_blackout": ttb, "attacker_failed": attacker_failed},
        "anchor_dn_under_same_attacks": {"survived": a_surv,
                                         "reward": anchor["cum_reward"],
                                         "game_over": anchor["game_over"],
                                         "n_trips": anchor["n_trips"],
                                         "n_attacks_in_schedule": len(schedule)},
        "time_to_blackout": ttb,
        "verdict": {
            "game_over": go, "time_to_blackout": ttb,
            "defender_survived": d_surv, "anchor_survived": a_surv,
            "defender_vs_anchor": cmp,
            "n_attacks": atk["n_calls"],
            "summary": (f"Blackout at t={ttb}" if go else f"No blackout (grid reached t={d_surv}/{horizon}) "
                        f"under {atk['n_calls']} attacks; "
                        f"LLM defender survived {d_surv} vs do-nothing {a_surv} under the SAME attacks.")},
        },
        "notes": notes,
    }
    json.dump(report, open(os.path.join(outdir, "report.json"), "w"), indent=2)
    print(f"  defender: survived {d_surv}/{horizon} cum {sim.get('cum_reward', 0):.0f} "
          f"go={go} trips={sim.get('n_trips', 0)} tok={(terminal.get('llm', {}) or {}).get('tokens_in')} "
          f"wall {wall_d:.0f}s")
    print(f"  attacker: {atk['n_calls']} attacks, wall {wall_a:.0f}s, "
          f"t2b={'t=%d' % ttb if go else 'none (no blackout)'}")
    print(f"  DN anchor under same {len(schedule)} attack-windows: survived {a_surv}/{horizon} "
          f"cum {anchor['cum_reward']:.0f} go={anchor['game_over']}")
    print(f"  verdict: {cmp} — {report['verdict']['summary']}")
    print(f"wrote {outdir}/report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
