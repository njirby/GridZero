"""simctl against the REAL backend (uvicorn + real grid2op SimSession) — the mock
tests in test_cli.py can't catch backend/CLI shape drift. Ports 18731-18740."""
import json
import urllib.error
import os
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI = os.path.join(ROOT, "cli", "simctl")
UVICORN = [sys.executable, "-m", "uvicorn"]
TOKEN = "testtok"


def _start(port, **extra_env):
    env = dict(os.environ, OPENCODE_DISABLE="1", SIMCTL_BACKEND_PORT=str(port),
               SIM_API_TOKEN=TOKEN, **extra_env)
    proc = subprocess.Popen(UVICORN + ["backend.app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 90:
        try:
            if urllib.request.urlopen(base + "/sim/status", timeout=2).status == 200:
                return proc, base
        except Exception:
            time.sleep(0.5)
    proc.kill()
    pytest.fail("backend did not start")


def _stop(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


@pytest.fixture(scope="module")
def real():
    proc, base = _start(18731)
    yield base
    _stop(proc)


@pytest.fixture(scope="module")
def short():  # horizon 3 -> reaches done quickly
    proc, base = _start(18732, SIM_HORIZON="3", SIM_CHRONIC="0")
    yield base
    _stop(proc)


@pytest.fixture(scope="module")
def norender():
    proc, base = _start(18733, RENDER_DISABLED="1")
    yield base
    _stop(proc)


def simctl(base, *args, token=None):
    env = dict(os.environ, SIM_API_URL=base)
    env.pop("SIM_API_TOKEN", None)  # the model never has the operator token
    if token:
        env["SIM_API_TOKEN"] = token
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True,
                          env=env, timeout=90)


def test_step_prints_real_values(real):
    simctl(real, "act", '{"set_line_status": {"0_4_1": -1}}')
    r = simctl(real, "step")
    assert r.returncode == 0
    assert "lines_down=1" in r.stdout  # was a hard-coded 0 before
    j = json.loads(simctl(real, "--json", "step").stdout)["data"]
    st = json.loads(simctl(real, "--json", "observe").stdout)["data"]
    assert j["lines_down"] == st["n_down"] >= 1
    assert j["overloads"] == [l["name"] for l in st["lines"] if l["rho"] > 1.0]


def test_illegal_reason_reaches_model(real):
    r = simctl(real, "act", '{"set_line_status": {"1_3_3": -1, "2_3_5": -1}}')
    assert r.returncode == 1
    assert "More than 1 line status" in r.stdout


def test_attacker_hidden_from_observe(real):
    simctl(real, "act", '{}')
    la = json.loads(simctl(real, "--json", "observe").stdout)["data"]["last_action"]
    assert la["source"] == "agent"


def test_observe_since_last_act_line(real):
    out = simctl(real, "act", '{}')
    assert out.returncode == 0
    r = simctl(real, "observe").stdout
    assert "since your last act" in r and ("no trips" in r or "tripped:" in r)


def test_event_and_state_gated(real):
    for path in ("/event", "/state", "/bench/stats"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(real + path, timeout=5)
        assert e.value.code == 403
    assert urllib.request.urlopen(real + "/state?token=" + TOKEN, timeout=5).status == 200


def test_post_done_rejected_nonzero(short):
    for _ in range(3):
        assert simctl(short, "step").returncode == 0
    assert "done=yes" in simctl(short, "status").stdout
    obs = json.loads(simctl(short, "--json", "observe").stdout)["data"]
    assert obs["done"] is True and obs["cause"] == "time_exceeded"
    for cmd in (("step",), ("act", "{}")):
        r = simctl(short, *cmd)
        assert r.returncode == 1
        assert "Episode is over (time_exceeded). Stop acting and write your summary." in r.stdout
        assert "reset" not in r.stdout


def test_render_disabled_clean(norender):
    r = simctl(norender, "render")
    assert r.returncode == 1
    assert r.stdout.strip() == "render is disabled in this environment"
    assert "Traceback" not in r.stderr


def test_render_escape_rejected(real):
    r = simctl(real, "render", "--out", "../../x.png")
    assert r.returncode == 1 and "invalid render filename" in r.stdout
