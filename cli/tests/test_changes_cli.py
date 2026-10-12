"""simctl observe --changes + SIM_ENV + SIM_HIDE_HORIZON against the REAL backend.
Ports 18741-18743.

The idf test prints the measured sizes (run with -s to see them):
chars/3 is the token estimate used in the report."""
import json
import urllib.request

import pytest

from test_real_backend import TOKEN, _start, _stop, simctl


@pytest.fixture(scope="module")
def case14():
    proc, base = _start(18741, SIM_CHRONIC="0", SIM_SEED="0")
    yield base
    _stop(proc)


@pytest.fixture(scope="module")
def idf():
    proc, base = _start(18742, SIM_ENV="l2rpn_idf_2023", SIM_CHRONIC="0", SIM_SEED="0")
    yield base
    _stop(proc)


@pytest.fixture(scope="module")
def hidden():  # horizon 3, hidden from the agent
    proc, base = _start(18743, SIM_HIDE_HORIZON="1", SIM_HORIZON="3", SIM_CHRONIC="0", SIM_SEED="0")
    yield base
    _stop(proc)


def _op_step(base, n):
    """Operator fast-forward (n<=50 per call) — the model can't do this."""
    req = urllib.request.Request(base + "/sim/step", data=json.dumps({"n": n}).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def test_default_env_is_case14(case14):
    assert "env=l2rpn_case14_sandbox" in simctl(case14, "status").stdout


def test_changes_cli_flow(case14):
    r = simctl(case14, "observe", "--changes")
    assert r.returncode == 0
    head, *rows = r.stdout.strip().splitlines()
    assert head.startswith("changes since start (full state, no previous view)")
    assert "lines 20/20" in head and "thresholds rho=0.01 mw=1.0" in head
    assert sum(1 for x in rows if x.startswith("line ")) == 20
    assert simctl(case14, "step").returncode == 0
    r = simctl(case14, "observe", "--changes")
    head = r.stdout.splitlines()[0]
    assert head.startswith("changes since t=0 -> t=1/")
    # JSON + threshold flags
    simctl(case14, "act", '{"set_line_status": {"0_4_1": -1}}')
    j = json.loads(simctl(case14, "--json", "observe", "--changes", "--rho-eps", "5", "--mw-eps", "1000").stdout)
    d = j["data"]
    assert d["thresholds"] == {"rho_eps": 5.0, "mw_eps": 1000.0}
    assert "0_4_1" in [l["name"] for l in d["lines"]] and d["gens"] == []
    assert "since your last act" in simctl(case14, "observe", "--changes").stdout
    simctl(case14, "act", '{"set_line_status": {"0_4_1": 1}}')


def test_detailed_is_a_baseline(case14):
    simctl(case14, "observe", "--detailed")
    t = json.loads(simctl(case14, "--json", "status").stdout)["data"]["t"]
    d = json.loads(simctl(case14, "--json", "observe", "--changes").stdout)["data"]
    assert d["changes_since_t"] == t and d["lines"] == []
    # the compact default observe is NOT a baseline
    simctl(case14, "step")
    simctl(case14, "observe")
    d = json.loads(simctl(case14, "--json", "observe", "--changes").stdout)["data"]
    assert d["changes_since_t"] == t


def test_changes_bad_args(case14):
    assert simctl(case14, "observe", "--changes", "--detailed").returncode == 3
    assert simctl(case14, "observe", "--rho-eps", "0.1").returncode == 3
    assert simctl(case14, "observe", "--changes", "--rho-eps", "-1").returncode == 3


def test_idf_env_sizes_and_attack(idf):
    assert "env=l2rpn_idf_2023" in simctl(idf, "status").stdout
    compact = simctl(idf, "observe")
    assert compact.returncode == 0 and "max_rho=" in compact.stdout
    det = simctl(idf, "observe", "--detailed")
    s = json.loads(det.stdout)
    assert s["env"] == "l2rpn_idf_2023" and len(s["lines"]) == 186 and len(s["subs"]) == 118
    # line name <or>_<ex>_<id> matches the or/ex endpoints -> adjacency is readable
    assert all(l["name"].split("_")[:2] == [l["or"][4:], l["ex"][4:]] for l in s["lines"])
    sizes = {"detailed": len(det.stdout), "observe(compact)": len(compact.stdout)}

    def changes(tag):
        h = simctl(idf, "observe", "--changes")
        assert h.returncode == 0
        sizes[f"changes {tag} (human)"] = len(h.stdout)
        return h.stdout

    simctl(idf, "step")
    changes("calm step")
    simctl(idf, "act", '{"redispatch": {"gen_14_8": 5.0}}')
    changes("after redispatch act")
    # chronic 0 / seed 0: do-nothing trips 3 lines at t=4 (discrete -> must appear)
    while json.loads(simctl(idf, "--json", "status").stdout)["data"]["t"] < 4:
        simctl(idf, "step")
    trip = changes("t=4 trips")
    for n in ("21_22_93", "26_31_106", "4_10_162"):
        assert f"line {n} " in trip and "status=down" in trip.split(f"line {n} ")[1].splitlines()[0]
    # built-in opponent: GeometricOpponentMultiArea cuts 99_105_62 at t=99 on this seed
    _op_step(idf, 50)
    _op_step(idf, 98 - 54)
    simctl(idf, "observe", "--detailed")
    assert json.loads(simctl(idf, "--json", "status").stdout)["data"]["t"] == 98
    st = json.loads(simctl(idf, "--json", "step").stdout)
    assert st["verbose"]["opponent_attack_line"] == ["99_105_62"]
    atk = changes("t=99 opponent attack")
    assert "line 99_105_62 " in atk and "status=down" in atk.split("line 99_105_62 ")[1].splitlines()[0]
    # JSON sizes (--json is compact JSON; a separate token'd viewer keeps the model's baseline)
    j_full = simctl(idf, "--json", "observe", "--changes", token="sizer")
    simctl(idf, "step")
    j_calm = simctl(idf, "--json", "observe", "--changes", token="sizer")
    sizes["changes full dump (json)"] = len(j_full.stdout)
    sizes["changes calm step (json)"] = len(j_calm.stdout)
    print("\nidf_2023 sizes (chars, ~tokens=chars/3):")
    for k, v in sizes.items():
        print(f"  {k:32s} {v:6d} chars  ~{v // 3:5d} tok")
    assert sizes["changes calm step (human)"] < sizes["detailed"] / 2


def test_horizon_shown_by_default(case14):
    assert "t=" in simctl(case14, "status").stdout and "/8064" in simctl(case14, "status").stdout
    assert "max_t" in json.loads(simctl(case14, "observe", "--detailed").stdout)


def test_hidden_horizon(hidden):
    st = simctl(hidden, "status").stdout
    assert "t=0 ·" in st and "/3" not in st
    assert "max_t" not in json.loads(simctl(hidden, "--json", "status").stdout)["data"]
    assert "max_t" not in json.loads(simctl(hidden, "observe", "--detailed").stdout)
    assert "max_t" not in json.loads(simctl(hidden, "--json", "observe").stdout)["data"]
    ch = simctl(hidden, "observe", "--changes").stdout
    assert "-> t=0 ·" in ch and "/3" not in ch
    assert "max_t" not in json.loads(simctl(hidden, "--json", "observe", "--changes").stdout)["data"]
    for _ in range(3):
        out = json.loads(simctl(hidden, "--json", "step").stdout)["data"]
        assert "max_t" not in out
    st = simctl(hidden, "status").stdout
    assert "done=yes" in st and "t=3 ·" in st
    s = json.loads(simctl(hidden, "observe", "--detailed").stdout)
    assert s["done"] is True and s["cause"] == "time_exceeded" and "max_t" not in s
    r = simctl(hidden, "step")
    assert r.returncode == 1 and "Episode is over (time_exceeded)" in r.stdout
    # the operator still sees the horizon
    req = urllib.request.Request(hidden + "/state", headers={"Authorization": "Bearer " + TOKEN})
    assert json.loads(urllib.request.urlopen(req, timeout=10).read())["sim"]["max_t"] == 3
