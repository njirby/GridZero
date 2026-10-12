"""Illegal / ambiguous / malformed action errors are factual and say whether time
advanced — on case14 and idf_2023. Also the per-element rolling history."""
import pytest

from backend.app.sim_session import HISTORY_MAX, SimSession

STEPPED = "Not applied: the step advanced to t={t} with do-nothing instead (reward 0)."


def _sess(env, chronic=5, steps=0):
    s = SimSession(env_name=env)
    s.reset(seed=0, options={"time serie id": chronic})
    for _ in range(steps):
        s.step(1)
    return s


def _act(s, action):
    t0 = s.latest_state()["t"]
    out, _, err = s.act(action)
    return t0, out["t"], err


def _stepped(s, action, kind, why):
    t0, t1, err = _act(s, action)
    assert t1 == t0 + 1, err
    assert err == f"{kind}: {why}. " + STEPPED.format(t=t1)


def _not_built(s, action, why):
    t0, t1, err = _act(s, action)
    assert t1 == t0
    assert err == f"Invalid action: {why}. Not applied; time did NOT advance (t={t0})."


@pytest.fixture(scope="module")
def c14():
    return _sess("l2rpn_case14_sandbox")


@pytest.fixture(scope="module")
def idf():
    return _sess("l2rpn_idf_2023")


# ---- case14 ----
def test_c14_not_redispatchable(c14):
    _stepped(c14, {"redispatch": {"gen_5_2": 2.0}}, "Ambiguous action",
             "gen_5_2 is not redispatchable (redispatchable gens: `simctl list gens --where redispatchable=true`)")


def test_c14_ramps(c14):
    _stepped(c14, {"redispatch": {"gen_1_0": -20.0}}, "Ambiguous action",
             "redispatch gen_1_0 -20 MW exceeds its ramp-down limit 5 MW/step (current target_dispatch 0 MW)")
    _stepped(c14, {"redispatch": {"gen_1_0": 20.0}}, "Ambiguous action",
             "redispatch gen_1_0 +20 MW exceeds its ramp-up limit 5 MW/step (current target_dispatch 0 MW)")


def test_c14_too_many_line_changes(c14):
    _stepped(c14, {"set_line_status": {"1_3_3": -1, "2_3_5": -1}}, "Illegal action",
             "at most 1 line status change per action (got 2: 1_3_3, 2_3_5)")


def test_c14_too_many_subs(c14):
    _stepped(c14, {"change_bus": {"lines_or_id": ["3_4_6"], "lines_ex_id": ["1_4_4"]}}, "Illegal action",
             "at most 1 substation change per action (got 2: sub_3, sub_4)")


def test_c14_curtail_non_renewable(c14):
    _stepped(c14, {"curtail": {"gen_1_0": 0.5}}, "Ambiguous action",
             "gen_1_0 is not renewable; only renewable gens can be curtailed "
             "(`simctl list gens --where renewable=true`)")


def test_c14_malformed(c14):
    _not_built(c14, {"redispatch": {"gen_9_9": 2.0}},
               "unknown generator 'gen_9_9'; closest names: gen_7_4, gen_5_3, gen_5_2 (`simctl list gens` lists them)")
    _not_built(c14, {"set_line_status": {"0_4_9": -1}},
               "unknown line '0_4_9'; closest names: 0_4_1, 3_4_6, 1_4_4 (`simctl list lines` lists them)")
    _not_built(c14, {"curtail": {"gen_5_2": 1.5}},
               "curtail value for gen_5_2 must be between 0.0 and 1.0 (fraction of pmax), got 1.5")
    _not_built(c14, {"set_bus": {"lines_or_id": {"3_4_6": 3}}},
               "set_bus.lines_or_id value for 3_4_6 must be 1 or 2 (busbar), -1 (disconnect) or 0, got 3")
    _not_built(c14, {"redispatch": {"gen_0_5": "abc"}},
               "`redispatch` value for gen_0_5 must be a number, got 'abc'")
    _not_built(c14, {"set_line_status": ["0_4_1"]},
               "`set_line_status` must be an object {\"<line name>\": value}, got list")
    _not_built(c14, {"set_storage": {"x": 1}}, "this grid has no storage units (got 'x')")
    _not_built(c14, {"change_bus": {"lines_xx": ["0_4_1"]}},
               "`change_bus` has unknown sub-key 'lines_xx' (valid: lines_or_id, lines_ex_id, loads_id, "
               "generators_id, storages_id)")


def test_c14_pmax_margin():
    s = _sess("l2rpn_case14_sandbox")
    for _ in range(6):                       # +90 MW target in 15 MW ramp steps
        assert _act(s, {"redispatch": {"gen_0_5": 15.0}})[2] is None
    _stepped(s, {"redispatch": {"gen_0_5": 15.0}}, "Illegal action",
             "redispatch gen_0_5 +15 MW would make its target_dispatch 105 MW, above pmax - pmin = 100 MW "
             "(pmax 100, pmin 0)")


# ---- idf_2023 ----
def test_idf_ramp_and_renewable(idf):
    _stepped(idf, {"redispatch": {"gen_7_2": -60.0}}, "Ambiguous action",
             "redispatch gen_7_2 -60 MW exceeds its ramp-down limit 3.5 MW/step (current target_dispatch 0 MW)")
    _stepped(idf, {"redispatch": {"gen_3_0": 1.0}}, "Ambiguous action",
             "gen_3_0 is not redispatchable (redispatchable gens: `simctl list gens --where redispatchable=true`)")
    _stepped(idf, {"curtail": {"gen_7_2": 0.5}}, "Ambiguous action",
             "gen_7_2 is not renewable; only renewable gens can be curtailed "
             "(`simctl list gens --where renewable=true`)")


def test_idf_storage(idf):
    _stepped(idf, {"set_storage": {"storage_22_0": 50.0}}, "Ambiguous action",
             "set_storage storage_22_0 50 MW exceeds its max charging power 12 MW (positive = charge)")
    _stepped(idf, {"set_storage": {"storage_22_0": -50.0}}, "Ambiguous action",
             "set_storage storage_22_0 -50 MW exceeds its max discharging power 12 MW (negative = discharge)")


def test_idf_line_cooldown_after_action(idf):
    assert _act(idf, {"set_line_status": {"0_1_0": -1}})[2] is None
    _stepped(idf, {"set_line_status": {"0_1_0": 1}}, "Illegal action",
             "line 0_1_0 cannot be changed for 3 more steps (cooldown: 3 steps after an action on it, 12 after a trip)")


def test_idf_sub_cooldown():
    s = _sess("l2rpn_idf_2023")
    assert _act(s, {"change_bus": {"lines_or_id": ["0_1_0"]}})[2] is None
    _stepped(s, {"change_bus": {"lines_or_id": ["0_1_0"]}}, "Illegal action",
             "substation sub_0 cannot be changed for 3 more steps (cooldown: 3 steps after a topology change)")


def test_idf_tripped_line_and_area_rule():
    s = _sess("l2rpn_idf_2023", chronic=0, steps=4)   # 3 lines trip at t=4 on chronic 0
    _stepped(s, {"set_line_status": {"4_10_162": 1}}, "Illegal action",
             "line 4_10_162 cannot be changed for 12 more steps (cooldown: 3 steps after an action on it, 12 after a trip)")
    _stepped(s, {"set_line_status": {"26_31_106": 1, "21_22_93": 1}}, "Illegal action",
             "at most 1 line status change per action in each area (got 2: 21_22_93, 26_31_106)")


def test_idf_malformed(idf):
    _not_built(idf, {"redispatch": {"gen_7_3": 1}},
               "unknown generator 'gen_7_3'; closest names: gen_79_43, gen_9_3, gen_7_2 (`simctl list gens` lists them)")
    _not_built(idf, {"set_storage": {"storage_22_9": 1.0}},
               "unknown storage unit 'storage_22_9'; closest names: storage_22_0, storage_112_6, storage_95_5")


# ---- history ----
def test_history_rows_and_cap():
    s = _sess("l2rpn_case14_sandbox")
    for _ in range(3):
        s.step(1)
    h = s.history("line", "1_4_4", last=2)
    assert [r["t"] for r in h["rows"]] == [2, 3] and set(h["rows"][0]) == {"t", "rho", "p_or", "status"}
    assert s.history("sub", "4")["name"] == "sub_4"
    assert len(s.history("gen", "gen_1_0", last=10_000)["rows"]) == 4   # t=0..3
    with pytest.raises(KeyError, match="closest: 1_4_4"):
        s.history("line", "1_4_5")
    s._hist.extend([s._hist[-1]] * (HISTORY_MAX + 10))
    assert len(s._hist) == HISTORY_MAX
    s.reset(seed=0)
    assert [r["t"] for r in s.history("line", "0_1_0")["rows"]] == [0]


def test_rejected_act_marked_in_last_action(c14):
    _act(c14, {"redispatch": {"gen_1_0": 50.0}})
    la = c14.latest_state()["last_action"]
    assert la["summary"] == "redispatch gen_1_0=50.0 (rejected, not applied)"
    _act(c14, {})
    assert "rejected" not in c14.latest_state()["last_action"]["summary"]
