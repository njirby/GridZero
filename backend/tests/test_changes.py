"""observe --changes: GET /sim/changes + SimSession.changes + c3_changes rules."""
import json, os
import jsonschema
import pytest
from referencing import Registry, Resource

from backend.app import c3_changes
from backend.app.sim_session import SimSession

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C3 = json.load(open(os.path.join(ROOT, "contracts", "C3-grid-state.schema.json")))
CH = json.load(open(os.path.join(ROOT, "contracts", "C3-grid-changes.schema.json")))
_VALIDATOR = jsonschema.Draft7Validator(
    CH, registry=Registry().with_resources([(C3["$id"], Resource.from_contents(C3)),
                                             (CH["$id"], Resource.from_contents(CH))]))


def _discrete_diff(old, new):
    return any(old[k] != new[k] for k in ("status", "overflow", "or", "ex", "or_bus", "ex_bus",
                                          "timestep_overflow")) or \
        ((old["cooldown"] == 0) != (new["cooldown"] == 0)) or new["cooldown"] > old["cooldown"]


@pytest.fixture()
def sess():
    s = SimSession()
    s.reset(seed=0, options={"time serie id": 0})
    return s


def test_first_call_is_full_dump(sess):
    d = sess.changes()
    _VALIDATOR.validate(d)
    assert d["full"] is True and d["changes_since_t"] is None
    assert d["lines_changed"] == d["n_line"] == 20 and len(d["lines"]) == 20
    assert d["subs_changed"] == 14 and d["gens_changed"] == 6
    assert d["thresholds"] == {"rho_eps": 0.01, "mw_eps": 1.0}
    # second call at the same t: nothing changed
    d2 = sess.changes()
    assert d2["full"] is False and d2["changes_since_t"] == 0
    assert d2["lines"] == [] and d2["subs"] == [] and d2["gens"] == []


def test_noop_step_reports_only_threshold_crossings(sess):
    base = sess.view_full()  # a --detailed view sets the baseline
    sess.step(1)
    cur = sess.latest_state()
    d = sess.changes()
    _VALIDATOR.validate(d)
    assert d["full"] is False and d["changes_since_t"] == 0 and d["t"] == 1
    reported = {l["id"] for l in d["lines"]}
    for old, new in zip(base["lines"], cur["lines"]):
        should = abs(new["rho"] - old["rho"]) >= 0.01 or _discrete_diff(old, new)
        assert (new["id"] in reported) == should, (old, new)
    for kind, fields in (("gens", ("p", "target_dispatch", "actual_dispatch")), ("subs", ("p",))):
        rep = {e["id"] for e in d[kind]}
        for old, new in zip(base[kind], cur[kind]):
            moved = any(abs(new[f] - old[f]) >= 1.0 for f in fields)
            if kind == "gens":
                moved = moved or old["connected"] != new["connected"]
            assert (new["id"] in rep) == moved
    # reported entities are complete and current
    by_id = {l["id"]: l for l in cur["lines"]}
    assert all(l == by_id[l["id"]] for l in d["lines"])
    assert 0 < d["lines_changed"] < 20  # a calm step is neither empty nor everything


def test_opening_a_line_reports_it_and_the_rerouted_flows(sess):
    sess.view_full()
    out, _, err = sess.act({"set_line_status": {"0_4_1": -1}})
    assert err is None
    d = sess.changes()
    names = {l["name"]: l for l in d["lines"]}
    assert names["0_4_1"]["status"] == "down"
    # the flow moved onto parallel paths: more than just the opened line changed
    assert len(names) > 1
    assert d["n_down"] == 1


def test_discrete_change_reported_below_thresholds(sess):
    base = sess.view_full()
    sess.act({"set_line_status": {"0_4_1": -1}})
    cur = sess.latest_state()
    d = sess.changes(rho_eps=100.0, mw_eps=1e9)  # continuous changes can't trigger
    # exactly the lines with a discrete change: the opened line (status) plus any
    # line whose overflow flag flipped (1_4_4 goes above 1.0 on this chronic)
    expect = {n["name"] for o, n in zip(base["lines"], cur["lines"]) if _discrete_diff(o, n)}
    assert {l["name"] for l in d["lines"]} == expect and "0_4_1" in expect
    assert d["subs"] == [] and d["gens"] == []
    assert d["thresholds"] == {"rho_eps": 100.0, "mw_eps": 1e9}
    sess.act({"set_line_status": {"0_4_1": 1}})
    d = sess.changes(rho_eps=100.0, mw_eps=1e9)
    assert ("0_4_1", "up") in [(l["name"], l["status"]) for l in d["lines"]]


def test_thresholds_overridable(sess):
    # same step, two viewers (separate baselines) asking with different thresholds
    sess.view_full("a")
    sess.view_full("b")
    sess.step(1)
    loose = sess.changes("a", rho_eps=0.0, mw_eps=0.0)
    tight = sess.changes("b", rho_eps=0.5, mw_eps=500.0)
    default = sess.changes("c")  # fresh viewer -> full
    assert tight["lines_changed"] == 0 and tight["gens_changed"] == 0
    assert loose["lines_changed"] > 0 and loose["thresholds"]["rho_eps"] == 0.0
    assert default["full"] is True


def test_viewers_have_separate_baselines(sess):
    sess.view_full("defender")
    sess.step(1)
    assert sess.changes("attacker")["full"] is True       # attacker never looked
    assert sess.changes("defender")["full"] is False


def test_reset_clears_baseline(sess):
    sess.view_full()
    sess.step(1)
    sess.reset(seed=0, options={"time serie id": 0})
    assert sess.changes()["full"] is True


# ---- pure rule tests on synthetic C3s ----
def _c3(t, lines, gens=(), subs=()):
    return {"t": t, "lines": list(lines), "gens": list(gens), "subs": list(subs)}


def _line(rho, cooldown=0, maint=-1, status="up", p_or=10.0):
    return {"id": 0, "name": "0_1_0", "or": "sub_0", "ex": "sub_1", "rho": rho, "p_or": p_or,
            "p_ex": -p_or, "status": status, "overflow": rho > 1.0, "cooldown": cooldown, "maint": maint}


def test_drift_accumulates_against_last_shown_value():
    _, v = c3_changes.changes(_c3(0, [_line(0.500)]), None)
    d, v = c3_changes.changes(_c3(1, [_line(0.505)]), v)
    assert d["lines"] == []                     # +0.005: below eps
    d, v = c3_changes.changes(_c3(2, [_line(0.511)]), v)
    assert len(d["lines"]) == 1                 # +0.011 vs the 0.500 the viewer saw
    d, v = c3_changes.changes(_c3(3, [_line(0.515)]), v)
    assert d["lines"] == []                     # baseline is now 0.511


def test_countdowns_only_report_transitions():
    _, v = c3_changes.changes(_c3(0, [_line(0.5, cooldown=0)]), None)
    d, v = c3_changes.changes(_c3(1, [_line(0.5, cooldown=3)]), v)
    assert len(d["lines"]) == 1                 # cooldown started
    d, v = c3_changes.changes(_c3(2, [_line(0.5, cooldown=2)]), v)
    assert d["lines"] == []                     # ticking down
    d, v = c3_changes.changes(_c3(3, [_line(0.5, cooldown=0)]), v)
    assert len(d["lines"]) == 1                 # cooldown ended
    d, v = c3_changes.changes(_c3(4, [_line(0.5, maint=12)]), v)
    assert len(d["lines"]) == 1                 # maintenance scheduled
    d, v = c3_changes.changes(_c3(5, [_line(0.5, maint=11)]), v)
    assert d["lines"] == []
    d, v = c3_changes.changes(_c3(6, [_line(0.5, maint=0)]), v)
    assert len(d["lines"]) == 1                 # maintenance now


def test_line_flow_mw_alone_does_not_trigger():
    _, v = c3_changes.changes(_c3(0, [_line(0.5, p_or=10.0)]), None)
    d, _ = c3_changes.changes(_c3(1, [_line(0.5, p_or=30.0)]), v)
    assert d["lines"] == []


# ---- HTTP route ----
def test_route_changes_and_full_view(client):
    full = client.get("/sim/state?view=full").json()
    assert full["ok"] and full["data"]["lines"]
    d = client.get("/sim/changes").json()
    assert d["ok"] and d["data"]["full"] is False
    assert d["data"]["changes_since_t"] == full["data"]["t"]
    _VALIDATOR.validate(d["data"])
    d = client.get("/sim/changes?rho_eps=0.5&mw_eps=50").json()
    assert d["data"]["thresholds"] == {"rho_eps": 0.5, "mw_eps": 50.0}
    assert client.get("/sim/changes?rho_eps=-1").status_code == 400


# ---- optional C3 fields (buses, overflow counter, dispatch) ----
def test_new_c3_fields_track_the_env(sess):
    c3 = sess.latest_state()
    assert c3["overflow_allowed"] == 2
    l0 = c3["lines"][0]
    assert (l0["or_bus"], l0["ex_bus"], l0["timestep_overflow"]) == (1, 1, 0)
    g = c3["gens"][0]
    assert g["connected"] is True and g["target_dispatch"] == 0.0 and g["actual_dispatch"] == 0.0
    sess.act({"set_line_status": {"0_4_1": -1}})          # 1_4_4 goes > 1.0 on this chronic
    c3 = sess.latest_state()
    by = {l["name"]: l for l in c3["lines"]}
    assert (by["0_4_1"]["or_bus"], by["0_4_1"]["ex_bus"]) == (-1, -1)
    assert by["1_4_4"]["overflow"] and by["1_4_4"]["timestep_overflow"] == 1
    sess.act({"redispatch": {"gen_1_0": -3.0}})
    g = {g["name"]: g for g in sess.latest_state()["gens"]}["gen_1_0"]
    assert g["target_dispatch"] == -3.0 and g["actual_dispatch"] != 0.0


def test_bus_move_and_overflow_counter_are_discrete(sess):
    sess.view_full()
    sess.act({"change_bus": {"lines_or_id": ["3_4_6"]}})
    d = sess.changes(rho_eps=100.0, mw_eps=1e9)
    moved = {l["name"]: l for l in d["lines"]}
    assert moved["3_4_6"]["or_bus"] == 2


def test_target_dispatch_uses_mw_threshold(sess):
    sess.view_full()
    sess.act({"redispatch": {"gen_1_0": -3.0}})
    hit = sess.changes(rho_eps=100.0, mw_eps=1.0)
    assert "gen_1_0" in [g["name"] for g in hit["gens"]]
    sess.view_full("x")
    sess.act({"redispatch": {"gen_1_0": -3.0}})       # cumulative -6
    miss = sess.changes("x", rho_eps=100.0, mw_eps=1e9)
    assert miss["gens"] == []


def test_overflow_counter_change_reported(sess):
    sess.act({"set_line_status": {"0_4_1": -1}})          # 1_4_4 overloaded, counter 1
    sess.view_full()
    sess.step(1)
    l = {l["name"]: l for l in sess.latest_state()["lines"]}["1_4_4"]
    assert l["timestep_overflow"] == 2                    # still overloaded, not yet tripped
    d = sess.changes(rho_eps=100.0, mw_eps=1e9)
    assert "1_4_4" in [x["name"] for x in d["lines"]]


# ---- hidden horizon (SIM_HIDE_HORIZON=1) ----
def test_hidden_horizon_routes(client, monkeypatch):
    from backend.app import main
    for path in ("/sim/status", "/sim/state", "/sim/state?view=full", "/sim/changes"):
        assert "max_t" in client.get(path).json()["data"], path
    monkeypatch.setattr(main, "HIDE_HORIZON", True)
    for path in ("/sim/status", "/sim/state", "/sim/state?view=full", "/sim/changes"):
        d = client.get(path).json()["data"]
        assert "max_t" not in d and "done" in d and "t" in d, path
    st = client.get("/sim/state").json()["data"]
    _VALIDATOR.validate(client.get("/sim/changes").json()["data"])
    jsonschema.validate(st, C3)
    out = client.post("/sim/step", json={}).json()["data"]
    assert "max_t" not in out
    # operator view keeps it
    assert "max_t" in client.get("/state").json()["sim"]


def test_default_thresholds_from_backend_env(client, monkeypatch):
    from backend.app import main
    monkeypatch.setattr(main, "CHANGES_RHO_EPS", 0.05)
    monkeypatch.setattr(main, "CHANGES_MW_EPS", 3.0)
    assert client.get("/sim/changes").json()["data"]["thresholds"] == {"rho_eps": 0.05, "mw_eps": 3.0}
    # an explicit query param still wins
    assert client.get("/sim/changes?rho_eps=0.2").json()["data"]["thresholds"] == {"rho_eps": 0.2, "mw_eps": 3.0}
