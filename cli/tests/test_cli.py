import json
import os
import signal
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI = os.path.join(ROOT, "cli", "simctl")
MOCK = os.path.join(ROOT, "contracts", "mock", "mock_sim_server.py")
PORT = 8733
BASE = f"http://127.0.0.1:{PORT}"


def _wait(url, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            import urllib.request
            if urllib.request.urlopen(url, timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(0.3)
    return False


@pytest.fixture(scope="session")
def mock():
    # Prefer the system python3 (mock is stdlib-only); fall back to the venv.
    py = sys.executable
    proc = subprocess.Popen([py, MOCK, "--port", str(PORT)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _wait(f"{BASE}/health"), "mock did not start"
        yield BASE
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def run_simctl(*args, base=None, as_json=False):
    cmd = [sys.executable, CLI]
    if as_json:
        cmd.append("--json")
    env = dict(os.environ)
    if base:
        env["SIM_API_URL"] = base
    cmd += list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    return r


def test_status(mock):
    r = run_simctl("status", base=mock)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sim up" in r.stdout and "t=0/8064" in r.stdout


def test_status_json(mock):
    r = run_simctl("status", base=mock, as_json=True)
    d = json.loads(r.stdout)
    assert d["ok"] is True and d["data"]["env"] == "l2rpn_case14_sandbox"


def test_step(mock):
    r = run_simctl("step", "50", base=mock)
    assert r.returncode == 0
    assert "stepped 50" in r.stdout and "t=50" in r.stdout


def test_act_ok(mock):
    r = run_simctl("act", '{"set_line_status":{"0_4_1":-1}}', base=mock)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "applied" in r.stdout


def test_act_illegal(mock):
    r = run_simctl("act", '{"set_line_status":{"99_99_9":-1}}', base=mock)
    assert r.returncode == 1, f"rc={r.returncode} out={r.stdout}"
    assert "rejected" in r.stdout or "Illegal" in r.stdout


def test_act_illegal_json(mock):
    r = run_simctl("act", '{"set_line_status":{"99_99_9":-1}}', base=mock, as_json=True)
    d = json.loads(r.stdout)
    assert d["ok"] is False and "error" in d and d["error"]


def test_act_bad_json(mock):
    r = run_simctl("act", "{not json", base=mock)
    assert r.returncode == 3
    assert "bad arguments" in r.stdout


def test_observe_compact(mock):
    r = run_simctl("observe", base=mock)
    assert r.returncode == 0
    assert "t=" in r.stdout and "top loads" in r.stdout and "max_rho" in r.stdout


def test_observe_detailed_is_c3(mock):
    r = run_simctl("observe", "--json", base=mock)
    d = json.loads(r.stdout)
    s = d["data"]
    for k in ("t", "max_t", "reward", "cum_reward", "lines", "subs", "n_line"):
        assert k in s, f"missing {k}"
    assert isinstance(s["lines"], list) and len(s["lines"]) == s["n_line"]


def test_render(mock):
    r = run_simctl("render", base=mock)
    assert r.returncode == 0
    path = r.stdout.split()[-2]  # "wrote <path> (800x500)"
    assert os.path.exists(path), f"render path missing: {path}"


def test_docs_lists_files():
    r = run_simctl("docs")
    assert r.returncode == 0
    assert "docs/" in r.stdout or "AGENTS.md" in r.stdout or "recipes/" in r.stdout


def test_unreachable():
    r = run_simctl("status", base="http://127.0.0.1:59999")
    assert r.returncode == 2
    assert "unreachable" in r.stdout


def test_attack_v0_stub(mock):
    r = run_simctl("attack", '{"line":"3_6_15","kind":"trip"}', base=mock)
    # mock has no /sim/attack route -> 404 envelope; simctl maps non-ok to a message
    assert r.stdout.strip()  # produced something
    assert r.returncode in (1, 2, 3)
