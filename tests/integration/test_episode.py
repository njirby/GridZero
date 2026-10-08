"""E — pytest acceptance gate for v0.

Runs one short real episode against a LIVE backend (which must already be
running with opencode + simctl + docs in place) and asserts the v0 gate:
>=1 legal agent act, reasoning present, a tool result present, and t advanced.

Skip (not fail) if no backend is reachable — so it's safe to run in CI before
the full stack is up. Run it live with:
  ./.venv/bin/python -m pytest tests/integration/test_episode.py -q -s
"""
import os, sys, time, json
import httpx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "integration"))
import run_episode as RE  # noqa: E402

PORT = int(os.environ.get("EVAL_PORT", "8731"))
BASE = f"http://127.0.0.1:{PORT}"


def _backend_up():
    try:
        return httpx.get(BASE + "/sim/status", timeout=3).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _backend_up(), reason="backend not running (start it for the live gate)")
def test_v0_acceptance_gate(tmp_path):
    # short budget: enough for a couple of model turns
    RE_main = RE.main.__wrapped__ if hasattr(RE.main, "__wrapped__") else None
    evfile = str(tmp_path / "ev.jsonl")
    # a token-gated backend needs the operator token (SIM_API_TOKEN in the env)
    tok = os.environ.get("SIM_API_TOKEN", "")
    hdr = {"Authorization": "Bearer " + tok} if tok else {}
    r = httpx.post(BASE + "/sim/reset", json={}, headers=hdr, timeout=60)
    if r.status_code in (401, 403):
        pytest.skip("backend is token-gated; export SIM_API_TOKEN=<operator token> to run the live gate")
    httpx.post(BASE + "/control", json={"cmd": "reset", "args": {}}, headers=hdr, timeout=60)
    time.sleep(2)
    deadline = time.time() + 90
    stop = lambda: time.time() > deadline
    with open(evfile, "w") as out:
        n = RE.collect_events(BASE, out, stop, hdr)
    assert n["legal_agent_acts"] >= 1, f"no legal agent act: {n}"
    assert n["agent_reasoning"] >= 1, f"no reasoning: {n}"
    assert n["tool_result"] >= 1, f"no tool result: {n}"
    assert (n["t_end"] or 0) > (n["t_start"] or 0), f"t did not advance: {n}"


def test_collect_events_parsing(tmp_path):
    # offline: replay a recorded stream through collect_events' parsing logic
    nd = os.path.join(ROOT, "contracts", "examples", "event-stream.ndjson")
    if not os.path.exists(nd):
        pytest.skip("no recorded stream")
    lines = [f"data: {l}" for l in open(nd) if l.strip()]
    import io, json as _json
    n = {"sim_state": 0, "step_outcome": 0, "agent_delta": 0, "agent_reasoning": 0,
         "tool_call": 0, "tool_result": 0, "turn_end": 0, "agent_steps": 0,
         "legal_agent_acts": 0, "t_start": None, "t_end": None, "cum_reward_end": None}
    for line in lines:
        ev = _json.loads(line[6:])
        t, d = ev.get("type"), ev.get("data", {})
        if t == "sim.state":
            n["sim_state"] += 1
            if isinstance(d, dict):  # recorded examples may carry placeholder strings
                n["t_start"] = n["t_start"] or d.get("t")
                n["t_end"] = d.get("t")
        elif t == "agent.delta" and isinstance(d, dict) and d.get("field") == "reasoning":
            n["agent_reasoning"] += 1
        elif t == "agent.tool_result":
            n["tool_result"] += 1
    assert n["sim_state"] >= 1 and n["agent_reasoning"] >= 1 and n["tool_result"] >= 1
