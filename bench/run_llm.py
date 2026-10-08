#!/usr/bin/env python
"""bench/run_llm.py — run ONE long-horizon benchmark episode with the LLM.

Robustness design (built for full-competition-length runs, e.g. 8064 steps):
  - UNLIMITED thinking time per step: no per-turn or per-episode deadline. The
    model may read docs, inspect state, or look around for as long as it needs
    on each step. Only a generous SAFETY CAP (default 12 h) guards against a
    wedged run; hitting it checkpoints (the trace is already on disk) and marks
    the episode `timeout`, it does not corrupt anything.
  - CHECKPOINT = append-only trace: the backend bus persists every event to
    runs/<ep>.jsonl as it happens. The sim is deterministic given
    (chronic, seed, action sequence), so the episode is fully auditable and the
    opencode session (which persists to disk) can be re-attached.
  - GENTLE auto-continue: if the sim makes NO progress for --poke-idle-s while
    not done, the runner sends a steer (queued at the model's next boundary, so
    it never interrupts a long think) reminding it to keep operating.

Each episode owns its own backend (one sim per process) + opencode session, on a
unique port, so episodes can run in parallel and one crash never affects others.

Usage:
  ./.venv/bin/python bench/run_llm.py --chronic 0 --horizon 288 --port 8801
  ./.venv/bin/python bench/run_llm.py --chronic 0 --horizon 8064 --port 8802 --repeats 3
Writes runs/llm-<chronic>-<ts>/episode-<i>.json (EpisodeResult) + backend log.
"""
import argparse, json, os, secrets, subprocess, sys, time, uuid, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import httpx
from bench.score import EpisodeResult
from bench.panel import load_panel

PY = os.path.join(ROOT, ".venv/bin/python")


def load_config(name):
    p = os.path.join(ROOT, "bench", "configs", f"{name}.json")
    if os.path.isfile(p):
        return json.load(open(p))
    print(f"  (no config {name}.json; using baseline defaults)")
    return {"name": name, "render": True, "docs": True, "observe_mode": "auto"}


def dn_attack_anchor(chronic, horizon, seed, attacks):
    """Do-nothing agent under the SAME deterministic attack schedule -> the fair
    adversarial anchor. Computed in-process on a fresh SimSession (free, ~tens of
    s) so the LLM defender result is compared against passive-under-identical-threat.
    Returns {survived, game_over, done, cum_reward, n_trips}."""
    import warnings
    warnings.filterwarnings("ignore")
    from backend.app.sim_session import SimSession
    s = SimSession()
    s.set_attack_schedule(attacks)
    s.reset(seed=seed, options={"time serie id": chronic, "max step": horizon})
    while not s.episode_stats().get("finished"):
        s.step(1)
        s.drain_opponent_events()
    st = s.episode_stats()
    return {"chronic": chronic, "horizon": horizon, "seed": seed, "n_attacks": len(attacks),
            "survived": st["t"], "game_over": st["game_over"], "done": st["done"],
            "cum_reward": st["cum_reward"], "n_trips": st["n_trips"]}


def kickoff_prompt(chronic, horizon, cfg=None, adversarial=False):
    cfg = cfg or {}
    render = cfg.get("render", True)
    docs = cfg.get("docs", True)
    obs_mode = cfg.get("observe_mode", "auto")
    obs_cmd = "simctl observe --detailed" if obs_mode == "detailed" else "simctl observe"
    obs_note = (" (ALWAYS use `simctl observe --detailed` for the full numeric state)"
                if obs_mode == "detailed" else "")
    tools = []
    if docs:
        tools.append("read docs (`simctl docs`, Read docs/)")
    tools.append("inspect state")
    if render:
        tools.append("render+Read the map (`simctl render` -> Read the PNG)")
    tools_txt = " or ".join(tools)
    adv_note = (
        f" WARNING: this is a DEFENSE scenario — other lines may FAIL (trip/disconnect) at any "
        f"time WITHOUT warning. Watch `simctl observe` for lines going down or new overloads you did "
        f"NOT cause, and recover quickly (re-route with change_bus/set_bus, redispatch, or re-open "
        f"tripped lines once their cooldown passes). You are being judged on how long the grid "
        f"survives under these failures and how well you keep reward up. "
        if adversarial else ""
    )
    return (
        f"BENCHMARK EPISODE — operate this power grid to completion. Horizon: {horizon} steps "
        f"(check progress with `simctl status`; it shows t/H). "
        f"Your task: keep the grid stable (no protection trips / cascades) and MAXIMIZE cumulative "
        f"reward. The reward explicitly pays you for using `redispatch` to cut line losses, so "
        f"actively redispatch generation to relieve the most loaded lines, not just to avoid trips. "
        f"Loop: `{obs_cmd}`{obs_note} -> decide (state your reasoning in 1-2 sentences) -> "
        f"`simctl act` -> verify the effect. Keep going, step by step, until `simctl status` "
        f"reports done=yes (t={horizon}) or the grid game-overs. {adv_note}"
        f"IMPORTANT: there is NO time limit — you may take as long as you need on each step to {tools_txt}. "
        f"Do not stop early and do not ask permission; keep operating until done. When done, print a "
        f"3-line summary of how you operated the grid. "
        f"HARD RULES: you CANNOT reset or restart the episode — it runs until done=yes or the grid goes "
        f"down, and a blackout ends the run (summarize and stop). Advancing time with no action (a no-op) "
        f"is `simctl step` (exactly 1 step) or `simctl act '{{}}'` — there is NO multi-step fast-forward "
        f"(no `simctl step N`); every step is your decision."
    )


def wait_up(base, path, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(base + path, timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_backend(port, acfg=None, model="qwen3.5-4b", net=None, model_url=None,
                  allow_no_netns=False):
    """Launch the episode backend. `net` (EpisodeNet) may be pre-created (the RL
    driver needs its gateway IP before the backend starts, to point opencode at
    the RL rollout endpoint). If netns setup fails the episode FAILS (RuntimeError)
    unless `allow_no_netns` (bench --allow-no-netns); OPENCODE_NETNS=0 also opts
    out. Returns (proc, operator_token, net); net is None iff NOT isolated.
    `model_url` overrides the agent's model endpoint
    (RL: the per-episode proxy; bench: the netns vLLM forwarder)."""
    acfg = acfg or {}
    log = open(os.path.join(ROOT, "runs", f"backend-bench-{port}.log"), "ab")
    env = dict(os.environ)
    env["SIMCTL_BACKEND_PORT"] = str(port)       # so simctl targets THIS backend
    env["OPENCODE_PORT"] = str(port + 200)       # unique opencode port per episode
    # operator token: the runner's /control + /sim/reset calls carry it; the
    # model's simctl never does (the sandbox strips it), so the model cannot
    # reset the episode, attack the grid, or drive /control.
    tok = secrets.token_hex(16)
    env["SIM_API_TOKEN"] = tok
    # per-episode netns (OPENCODE_NETNS=0 to disable): the backend runs inside
    # it; the sandboxed opencode joins it; the agent's only external service
    # is vLLM via the gateway-IP forwarder (no sibling backends, no internet).
    if net is None and os.environ.get("OPENCODE_NETNS", "1") == "1":
        try:
            from bench.netns import EpisodeNet
            net = EpisodeNet(port, log_dir=os.path.join(ROOT, "runs"))
            net.setup()
        except Exception as e:
            if net:
                net.teardown()
            net = None
            if not allow_no_netns:
                raise RuntimeError(f"netns setup failed ({e}); refusing to run without network "
                                   "isolation (pass --allow-no-netns to override)") from e
            print(f"  (netns setup failed: {e} — running WITHOUT network isolation)")
    if model_url:
        env["VLLM_FORWARD_URL"] = model_url      # RL: per-episode proxy
    elif net:
        env["VLLM_FORWARD_URL"] = net.vllm_url()
    env["OPENCODE_SANDBOX"] = "1"                # filesystem sandbox: hide the answer key
    env["OPENCODE_MODEL"] = model or acfg.get("model") or "qwen3.5-4b"
    if not acfg.get("render", True):
        env["RENDER_DISABLED"] = "1"             # no-vision ablation
    if not acfg.get("docs", True):
        env["OPENCODE_NO_DOCS"] = "1"            # no-docs ablation (sandbox builds ws w/o docs)
    if acfg.get("doc_warning"):
        env["OPENCODE_DOC_WARNING"] = acfg["doc_warning"]  # A/B: append a doc variant
    # Inside the netns the backend is reached via the veth IP, so bind 0.0.0.0;
    # the driver's sandboxed bwrap is a child of the backend and thus inherits
    # the netns automatically (no nsenter needed in bwrap_argv).
    host = "0.0.0.0" if net else "127.0.0.1"
    argv = [PY, "-m", "uvicorn", "backend.app.main:app", "--host", host, "--port", str(port)]
    if net:
        argv = net.ns(argv)
    p = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    # state file for watchers (scripts/bench_watch.py): the backend's reachable
    # address (netns IP when isolated) + operator token. 0600, removed at teardown.
    try:
        fd = os.open(state_path(port), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"base": net.backend_base() if net else f"http://127.0.0.1:{port}",
                       "token": tok, "isolated": net is not None}, f)
    except OSError as e:
        print(f"  (could not write backend state file: {e})")
    return p, tok, net


def state_path(port):
    return os.path.join(ROOT, "runs", f"backend-bench-{port}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chronic", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=None)  # default from panel
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--safety-cap-h", type=float, default=12.0)
    ap.add_argument("--poke-idle-s", type=int, default=180)
    ap.add_argument("--liveness-min", type=float, default=25.0,
                    help="fail the episode as agent_failed if the agent shows no activity this long")
    ap.add_argument("--stall-min", type=float, default=40.0,
                    help="if the model produces NO output (deltas/tool) this long, the session "
                         "is wedged -> restart the agent once (fresh session), then fail if it wedges again")
    ap.add_argument("--config", default="baseline", help="ablation config name (bench/configs/<name>.json)")
    # adversarial (robustness) mode: a deterministic scripted attacker fires while
    # the LLM defends (blind to who attacked); scored vs do-nothing under the SAME attacks.
    ap.add_argument("--adversarial", action="store_true", help="enable the scripted attacker")
    ap.add_argument("--attack-interval", type=int, default=96, help="attack every N sim-steps (72h=864; 96=1day)")
    ap.add_argument("--attack-duration", type=int, default=24, help="each attack lasts N sim-steps (24=2h)")
    ap.add_argument("--attack-seed", type=int, default=None, help="seed for the attack schedule (default: --seed)")
    ap.add_argument("--allow-no-netns", action="store_true",
                    help="continue WITHOUT network isolation if netns setup fails (result is flagged isolated=false)")
    a = ap.parse_args()
    panel = load_panel()
    horizon = a.horizon or panel["horizons"]["pilot"]
    a.cfg = load_config(a.config)
    # build the deterministic attack schedule + the free do-nothing-under-attack anchor
    if a.adversarial:
        from dataclasses import asdict
        from bench.attacker import generate_attacks, ATTACK_LINES
        a.attack_seed = a.attack_seed if a.attack_seed is not None else a.seed
        # convert Attack dataclasses -> dicts (the backend + anchor expect dicts)
        a.attacks = [asdict(x) for x in generate_attacks(
            seed=a.attack_seed, max_step=horizon, interval=a.attack_interval,
            duration=a.attack_duration, lines=ATTACK_LINES)]
        a.dn_anchor = dn_attack_anchor(a.chronic, horizon, a.seed, a.attacks)
    else:
        a.attacks = []
        a.dn_anchor = None
    from bench.panel import config_hash
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(ROOT, "runs", f"llm-{a.chronic}-{ts}")
    os.makedirs(outdir, exist_ok=True)
    cfg = {"model": a.model, "chronic": a.chronic, "horizon": horizon, "seed": a.seed,
           "repeats": a.repeats, "safety_cap_h": a.safety_cap_h, "ablation": a.config,
           "adversarial": a.adversarial,
           "attack_interval": a.attack_interval if a.adversarial else None,
           "attack_duration": a.attack_duration if a.adversarial else None,
           "n_attacks": len(a.attacks) if a.adversarial else 0,
           "dn_attack_anchor": a.dn_anchor,
           "config_hash": config_hash(
               model=a.model, horizon=horizon, panel=panel, ablation=a.cfg, seed=a.seed,
               adversarial=({"interval": a.attack_interval, "duration": a.attack_duration,
                             "seed": a.attack_seed} if a.adversarial else None)),
           "allow_no_netns": a.allow_no_netns, "outdir": outdir}
    json.dump(cfg, open(os.path.join(outdir, "config.json"), "w"), indent=2)
    if a.adversarial:
        json.dump({"dn_attack_anchor": a.dn_anchor,
                   "attacks": a.attacks},
                  open(os.path.join(outdir, "dn_anchor.json"), "w"), indent=2)
        print(f"  ADVERSARIAL: {len(a.attacks)} attacks (interval {a.attack_interval}, "
              f"dur {a.attack_duration}); DN-under-attack anchor: survived {a.dn_anchor['survived']}/{horizon} "
              f"cum {a.dn_anchor['cum_reward']:.0f} go={a.dn_anchor['game_over']}")
    results = []
    for i in range(a.repeats):
        res = run_episode(a.port, a, outdir, horizon)
        if res is None:
            return 2
        results.append(res)
        fn = os.path.join(outdir, f"episode-{i}.json")
        json.dump(res.to_dict(), open(fn, "w"), indent=2)
        print(f"  repeat {i}: survived {res.survived}/{horizon} cum {res.cum_reward:.0f} "
              f"done {res.done} trips {res.n_trips} tok {res.tokens_in}/{res.tokens_out} "
              f"${res.cost_usd} wall {res.wall_clock_s}s")
    json.dump([r.to_dict() for r in results], open(os.path.join(outdir, "results.json"), "w"), indent=2)
    print(f"wrote {outdir}/results.json ({len(results)} episodes)")
    return 0


def _hdr(tok):
    return {"Authorization": "Bearer " + tok} if tok else {}


def run_one(base, a, i, outdir, horizon, tok=""):
    t_start = time.time()
    # start the episode (pins sim + fresh opencode session + kickoff + attack schedule)
    r = httpx.post(base + "/bench/start",
                   json={"chronic": a.chronic, "horizon": horizon, "seed": a.seed,
                         "model": a.model,
                         "kickoff": kickoff_prompt(a.chronic, horizon, a.cfg, adversarial=a.adversarial),
                         "attacks": a.attacks},
                   headers=_hdr(tok), timeout=300)
    r.raise_for_status()
    ep = r.json()["data"]["ep"]
    cap = a.safety_cap_h * 3600
    liveness = a.liveness_min * 60.0
    stall = a.stall_min * 60.0
    now = time.time()
    last_t, last_t_wall = 0, now
    saw_agent_active = r.json()["data"].get("agent_active", False)
    max_agent_events = 0
    # wait for the agent to come alive (opencode may be booting); if it never does,
    # FAIL FAST as agent_failed instead of poking a dead agent for hours.
    while time.time() - t_start < liveness and not saw_agent_active:
        st = httpx.get(base + "/bench/stats", headers=_hdr(tok), timeout=30).json()
        saw_agent_active = st.get("agent_active", False)
        max_agent_events = max(max_agent_events, st.get("n_agent_events", 0))
        if st["sim"].get("done") or st["sim"].get("game_over"):
            break
        time.sleep(15)
    if not saw_agent_active and max_agent_events == 0:
        print(f"  chronic {a.chronic}: AGENT FAILED to start within {a.liveness_min} min -> agent_failed (aborting)")
        stats = httpx.get(base + "/bench/stats", headers=_hdr(tok), timeout=30).json()
        return build_result(a, i, horizon, stats, time.time() - t_start, ep,
                            notes="agent_failed(no agent activity)", agent_failed=True)
    restarted = False
    while True:
        now = time.time()
        st = httpx.get(base + "/bench/stats", headers=_hdr(tok), timeout=30).json()
        s = st["sim"]
        saw_agent_active = saw_agent_active or st.get("agent_active", False)
        max_agent_events = max(max_agent_events, st.get("n_agent_events", 0))
        t = s.get("t", 0)
        if t != last_t:
            last_t, last_t_wall = t, now
        lmo = st.get("last_model_output_ts", 0) or 0
        # how long since ANY real model output (deltas/tool) — a legitimately long
        # thinking turn streams deltas continuously, so this stays small; a WEDGED
        # session goes quiet forever.
        quiet = now - lmo if lmo else (now - t_start)
        # --- terminal: episode finished ---
        if s.get("done") or s.get("game_over"):
            # Build the result from the CAPTURED terminal state (`st`/`s` above).
            # Do NOT re-fetch: if the backend crashes in the teardown window (observed
            # on a game-over episode), a re-fetch reads a FRESH sim (t=0) and corrupts
            # the record. A defensive re-fetch is only used if it still shows the same
            # terminal episode at t >= the captured t.
            terminal = st
            try:
                time.sleep(8)  # let the model finish any in-flight summary turn
                st2 = httpx.get(base + "/bench/stats", headers=_hdr(tok), timeout=20).json()
                s2 = st2.get("sim", {})
                if (s2.get("t", -1) >= t and (s2.get("done") or s2.get("game_over"))
                        and st2.get("ep") == ep):
                    terminal = st2
            except Exception:
                pass  # backend gone -> use the captured terminal state
            agent_failed = (max_agent_events == 0 and t <= 0)
            return build_result(a, i, horizon, terminal, time.time() - t_start, ep,
                                notes=("agent_failed(no agent activity)" if agent_failed
                                       else "done" if s.get("done") else "game_over"),
                                agent_failed=agent_failed)
        # --- terminal: safety cap ---
        if now - t_start > cap:
            print(f"  SAFETY CAP hit at t={t}; checkpointing (trace on disk, resumable)")
            return build_result(a, i, horizon, st, time.time() - t_start, ep,
                                notes="timeout(safety cap)", agent_failed=not saw_agent_active)
        # --- STALL: no model output for stall_min -> the session wedged ---
        if quiet > stall:
            if not restarted:
                print(f"  t={t}/{horizon}: model quiet {int(quiet)}s (> {a.stall_min} min) -> restarting agent (fresh session)")
                httpx.post(base + "/control", json={"cmd": "restart_agent", "args": {}},
                           headers=_hdr(tok), timeout=60)
                restarted = True
                lmo = time.time()  # give the fresh session a full stall window
            else:
                print(f"  t={t}/{horizon}: STILL quiet after restart -> agent_failed (aborting, not burning more compute)")
                return build_result(a, i, horizon, st, time.time() - t_start, ep,
                                    notes="agent_failed(wedged after restart)", agent_failed=True)
            time.sleep(30)
            continue
        # --- gentle nudge: model is alive (producing output) but not stepping the sim ---
        if (now - last_t_wall) > a.poke_idle_s and quiet < (stall * 0.5):
            msg = (f"continue operating the grid: you are at t={t}/{horizon}. There is no time "
                   f"pressure — take the next step (observe, decide, act). Keep going until done.")
            httpx.post(base + "/control", json={"cmd": "instruction", "args": {"text": msg}},
                       headers=_hdr(tok), timeout=30)
            last_t_wall = now
            print(f"  t={t}/{horizon} no-step>{a.poke_idle_s}s (model active) -> poked continue")
        time.sleep(20)


def build_result(a, i, horizon, stats, wall, ep, notes="", agent_failed=False):
    sim = stats["sim"]
    llm = stats.get("llm", {})
    res = EpisodeResult(
        agent=a.model, chronic=a.chronic, horizon=horizon, survived=sim.get("t", 0),
        done=bool(sim.get("done")), cum_reward=float(sim.get("cum_reward", 0.0)),
        n_trips=sim.get("n_trips", 0), n_illegal=sim.get("n_illegal", 0),
        n_ambiguous=sim.get("n_ambiguous", 0), peak_rho=sim.get("peak_rho", 0.0),
        n_down_final=sim.get("n_down_final", 0), game_over=bool(sim.get("game_over")),
        tokens_in=llm.get("tokens_in"), tokens_out=llm.get("tokens_out"),
        reasoning_tokens=llm.get("reasoning_tokens"), cost_usd=llm.get("cost_usd"),
        wall_clock_s=round(wall, 1), ep=ep, notes=notes, agent_failed=bool(agent_failed),
        isolated=getattr(a, "isolated", None))
    # enrich with behavior from the append-only trace (acts/observes/renders, etc.)
    try:
        from bench.analyze_trace import analyze_trace
        trace = os.path.join(ROOT, "runs", f"{ep}.jsonl")
        if os.path.exists(trace):
            b = analyze_trace(trace)
            res.llm_turns = b["n_llm_turns"] or res.llm_turns
            res.simctl_acts = b["simctl_act_calls"]
            res.simctl_observes = b["simctl_observe_calls"]
            res.simctl_renders = b["simctl_render_calls"]
            res.notes = (res.notes + " | " if res.notes else "") + \
                f"actions:{b['agent_action_key_breakdown']}"
    except Exception:
        pass
    return res


def run_episode(port, a, outdir, horizon, net=None, model_url=None):
    """Full single-episode lifecycle: backend + boot wait + run_one + teardown.

    `a` is an argparse.Namespace with at least: model, cfg, chronic, seed,
    attacks, adversarial, safety_cap_h, liveness_min, stall_min, poke_idle_s.
    Returns EpisodeResult (None if the backend failed to boot). Reused by the
    RL rollout driver (rl/gridzero_agent.py), which calls this per episode on
    its own port so episodes run in parallel; it pre-creates `net` (to know the
    gateway IP) and passes `model_url` (the per-episode RL proxy)."""
    bp, tok, net = start_backend(port, a.cfg, model=a.model, net=net, model_url=model_url,
                                 allow_no_netns=getattr(a, "allow_no_netns", False))
    a.isolated = net is not None
    base = net.backend_base() if net else f"http://127.0.0.1:{port}"
    try:
        if not wait_up(base, "/sim/status"):
            print("ERROR: backend did not boot; see", os.path.join(ROOT, "runs", f"backend-bench-{port}.log"))
            return None
        return run_one(base, a, 0, outdir, horizon, tok)
    finally:
        # graceful shutdown (lets the backend's on_shutdown reap its sandbox via
        # oc.stop()); give it time, then hard-kill, then a port-based sweep as a
        # belt-and-suspenders (a SIGKILL'd backend can't finish its cleanup).
        bp.terminate()
        try:
            bp.wait(timeout=30)
        except Exception:
            bp.kill()
        try:
            os.remove(state_path(port))
        except OSError:
            pass
        try:
            from bench import sandbox
            sandbox.kill_sandbox_proc(None, oc_port=port + 200)
        except Exception as e:
            print(f"  (sandbox sweep after shutdown: {e})")
        if net:
            try:
                net.teardown()
            except Exception as e:
                print(f"  (netns teardown: {e})")


if __name__ == "__main__":
    sys.exit(main())
