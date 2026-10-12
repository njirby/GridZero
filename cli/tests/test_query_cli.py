"""simctl grid / get / list / history against the REAL backend (case14 + idf_2023),
including the completeness requirement: every (entity, field) of C3 is reachable
through `get` (text and --json). Ports 18744-18745."""
import json
import urllib.request

import pytest

from test_real_backend import TOKEN, _start, _stop, simctl

KINDS = {"line": "lines", "sub": "subs", "gen": "gens"}


@pytest.fixture(scope="module")
def case14():
    proc, base = _start(18744, SIM_CHRONIC="0", SIM_SEED="0")
    yield base
    _stop(proc)


@pytest.fixture(scope="module")
def idf():
    proc, base = _start(18745, SIM_ENV="l2rpn_idf_2023", SIM_CHRONIC="0", SIM_SEED="0")
    yield base
    _stop(proc)


def _c3(base):
    return json.loads(simctl(base, "--json", "observe").stdout)["data"]


def _v(v):
    return v if isinstance(v, str) else json.dumps(v)


def _blocks(text):
    """Parse `get` text: {"<kind> <name>": [field lines]}."""
    out, cur = {}, None
    for ln in text.splitlines():
        if not ln.startswith(" "):
            cur = ln.strip()
            out[cur] = []
        else:
            out[cur].append(ln)
    return out


def _assert_complete(base):
    s = _c3(base)
    for kind, key in KINDS.items():
        text = simctl(base, "get", f"{kind}.*")
        assert text.returncode == 0
        blocks = _blocks(text.stdout)
        js = json.loads(simctl(base, "--json", "get", f"{kind}.*").stdout)["data"]["objects"]
        assert js == s[key]
        for e in s[key]:
            body = blocks[f"{kind} {e['name']}"]
            for f, val in e.items():
                assert f"  {f}: {_v(val)}" in body, (kind, e["name"], f)
    blocks = _blocks(simctl(base, "get", "state").stdout)
    st = json.loads(simctl(base, "--json", "get", "state").stdout)["data"]["objects"][0]
    for f, val in s.items():
        if f in KINDS.values():
            continue
        assert f"  {f}: {_v(val)}" in blocks["state"], f
        assert st[f] == val


def test_completeness_case14(case14):
    simctl(case14, "act", '{"redispatch": {"gen_1_0": -3.0}}')
    _assert_complete(case14)


def test_completeness_idf(idf):
    simctl(idf, "act", '{"redispatch": {"gen_7_2": -3.5}}')
    _assert_complete(idf)


def test_grid_tables_and_flags(case14):
    simctl(case14, "act", '{"redispatch": {"gen_1_0": -2.0}}')
    simctl(case14, "act", '{"set_line_status": {"0_4_1": -1}}')
    g = simctl(case14, "grid")
    assert g.returncode == 0
    assert g.stdout == simctl(case14, "observe", "--detailed").stdout
    s = _c3(case14)
    rows = {ln.split()[0]: ln for ln in g.stdout.splitlines() if ln.startswith("  ")}
    assert "down" in rows["0_4_1"].split()
    for l in s["lines"]:   # flags appear exactly when the field is non-default
        assert ("overflow " in rows[l["name"]]) == (l["timestep_overflow"] > 0)
        if l["timestep_overflow"]:
            assert f"overflow {l['timestep_overflow']}/2" in rows[l["name"]]
        assert ("cooldown=" in rows[l["name"]]) == (l["cooldown"] > 0)
    gen = [x for x in s["gens"] if x["name"] == "gen_1_0"][0]
    assert f"dispatch target={gen['target_dispatch']:g} actual={gen['actual_dispatch']:g}" in rows["gen_1_0"]
    assert "dispatch" not in rows["gen_5_2"]
    assert rows["gen_1_0"].split()[3] == "R-" and rows["gen_5_2"].split()[3] == "-W"
    for sec in ("LINES (20)", "SUBS (14)", "GENS (6)"):
        assert sec in g.stdout
    simctl(case14, "act", '{"set_line_status": {"0_4_1": 1}}')
    simctl(case14, "act", '{"change_bus": {"lines_or_id": ["3_4_6"]}}')
    rows = {ln.split()[0]: ln for ln in simctl(case14, "grid").stdout.splitlines() if ln.startswith("  ")}
    assert "or_bus=2" in rows["3_4_6"]
    simctl(case14, "act", '{"change_bus": {"lines_or_id": ["3_4_6"]}}')


def test_get_sub_attached_and_aliases(case14):
    a = simctl(case14, "get", "sub.4").stdout
    assert a == simctl(case14, "get", "sub.sub_4").stdout
    assert "attached lines (" in a and "3_4_6 ex end, bus=" in a and "4_5_17 or end, bus=" in a
    j = json.loads(simctl(case14, "--json", "get", "sub.sub_1").stdout)["data"]
    assert [g["gen"] for g in j["attached"]["sub_1"]["gens"]] == ["gen_1_0"]
    assert simctl(case14, "get", "line.0").stdout.startswith("line 0_1_0")


def test_get_errors(case14):
    r = simctl(case14, "get", "gen.gen_1_1")
    assert r.returncode == 3 and "unknown gen 'gen_1_1'" in r.stdout and "gen_1_0" in r.stdout
    r = simctl(case14, "get", "bus.1")
    assert r.returncode == 3 and "line, sub, gen" in r.stdout


def test_list_sort_top_where(idf):
    s = _c3(idf)
    out = simctl(idf, "list", "lines", "--sort", "rho", "--top", "5").stdout.splitlines()
    names = [ln.split()[0] for ln in out[1:]]
    assert names == [l["name"] for l in sorted(s["lines"], key=lambda l: -l["rho"])[:5]]
    j = json.loads(simctl(idf, "--json", "list", "gens", "--where", "redispatchable=true").stdout)["data"]
    assert j["n_match"] == 20 and all(g["redispatchable"] for g in j["objects"])
    j = json.loads(simctl(idf, "--json", "list", "lines", "--where", "name=4_10_*").stdout)["data"]
    assert [l["name"] for l in j["objects"]] == ["4_10_162"]
    j = json.loads(simctl(idf, "--json", "list", "lines", "--where", "rho>0.5", "--where", "status=up").stdout)["data"]
    assert j["n_match"] == sum(1 for l in s["lines"] if l["rho"] > 0.5 and l["status"] == "up")
    # fields not visible in the compact row are appended
    assert "p_ex=" in simctl(idf, "list", "lines", "--sort", "p_ex", "--top", "1").stdout
    r = simctl(idf, "list", "lines", "--sort", "nope")
    assert r.returncode == 3 and "fields:" in r.stdout


def test_every_field_sortable_and_filterable(case14):
    s = _c3(case14)
    for kind, key in KINDS.items():
        first = s[key][0]
        for f, val in first.items():
            r = simctl(case14, "--json", "list", key, "--sort", f)
            assert r.returncode == 0, (kind, f, r.stdout)
            j = json.loads(simctl(case14, "--json", "list", key, "--where", f"{f}={_v(val)}").stdout)["data"]
            assert first["name"] in [e["name"] for e in j["objects"]], (kind, f)


def test_history(case14):
    for _ in range(3):
        simctl(case14, "step")
    t = _c3(case14)["t"]
    h = json.loads(simctl(case14, "--json", "history", "line.1_4_4", "--last", "3").stdout)["data"]
    assert [r["t"] for r in h["rows"]] == [t - 2, t - 1, t]
    assert h["fields"] == ["rho", "p_or", "status"]
    cur = [l for l in _c3(case14)["lines"] if l["name"] == "1_4_4"][0]
    assert abs(h["rows"][-1]["rho"] - cur["rho"]) < 1e-3
    g = json.loads(simctl(case14, "--json", "history", "gen.gen_1_0").stdout)["data"]
    assert g["fields"] == ["p", "target_dispatch", "actual_dispatch"] and len(g["rows"]) == min(12, t + 1)
    txt = simctl(case14, "history", "sub.4", "--last", "2").stdout
    assert txt.startswith("sub sub_4 — last 2 step(s)")
    r = simctl(case14, "history", "line.nope")
    assert r.returncode == 3 and "unknown line" in r.stdout
    # an operator reset clears the history
    req = urllib.request.Request(case14 + "/sim/reset", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN})
    urllib.request.urlopen(req, timeout=60).read()
    h = json.loads(simctl(case14, "--json", "history", "line.1_4_4").stdout)["data"]
    assert [r["t"] for r in h["rows"]] == [0]


def test_idf_sizes(idf):
    s = _c3(idf)
    g = simctl(idf, "grid").stdout
    print(f"\nidf grid: {len(g)} chars, {len(g.splitlines())} lines (~{len(g) // 3} tok); "
          f"old indent=2 JSON: {len(json.dumps(s, indent=2))} chars")
    assert len(g) < 20_000 and len(g.splitlines()) < 400
