"""Regression tests for the sim backend correctness/isolation fixes (REAL SimSession)."""
import os

import jsonschema
import pytest

from backend.app.sim_session import SimSession
import backend.app.main as m

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = __import__("json").load(open(os.path.join(ROOT, "contracts", "C3-grid-state.schema.json")))


@pytest.fixture()
def sess(tmp_path):
    s = SimSession(render_dir=str(tmp_path / "r"))
    s.reset(options={"max step": 6})
    return s


# ---- 1: step/act outcomes carry accurate lines_down / overloads
def test_step_outcome_lines_down_overloads(sess):
    out, verb = sess.step(1)
    assert out["lines_down"] == 0 and out["overloads"] == []
    out, verb, err = sess.act({"set_line_status": {"0_4_1": -1}})
    assert err is None
    assert out["lines_down"] == 1
    st = sess.latest_state()
    assert out["overloads"] == [l["name"] for l in st["lines"] if l["rho"] > 1.0]
    out2, _ = sess.step(1)
    assert out2["lines_down"] == sum(1 for l in sess.latest_state()["lines"] if l["status"] == "down")


# ---- 2: illegal reason is surfaced
def test_illegal_reason_in_error(sess):
    out, verb, err = sess.act({"set_line_status": {"0_4_1": -1, "1_3_3": -1}})
    assert out["illegal"] is True
    assert "More than 1 line status" in err and err.startswith("Illegal action")


# ---- 3/4: done, cause, post-done rejection, no "reset" hint
def test_done_cause_and_post_done_rejected(sess):
    for _ in range(6):
        sess.step(1)
    st = sess.latest_state()
    assert st["done"] is True and st["cause"] == "time_exceeded"
    assert sess.status()["done"] is True
    jsonschema.validate(st, SCHEMA)
    out, verb = sess.step(1)
    assert verb["rejected"].startswith("Episode is over (time_exceeded). Stop acting")
    assert "reset" not in verb["rejected"]
    out, verb, err = sess.act({})
    assert err == "Episode is over (time_exceeded). Stop acting and write your summary."


def test_game_over_cause(sess):
    sess._game_over = True
    sess._finished = True
    assert sess._cause() == "game_over"
    assert "game_over" in sess._over_msg() and "reset" not in sess._over_msg()


def test_observe_done_agrees_with_status_on_game_over(sess):
    sess._game_over = sess._finished = True
    sess._last_c3 = sess._build_c3(sess._obs, None, "")
    assert sess.latest_state()["done"] is True and sess.latest_state()["cause"] == "game_over"
    assert sess.status()["done"] is True


# ---- 5: render path confined
@pytest.mark.parametrize("bad", ["/tmp/x.png", "../../x.png", "a/b.png", "x.txt", "..", ".hidden.png"])
def test_render_rejects_bad_out(sess, bad):
    with pytest.raises(ValueError):
        sess.render(out=bad)


def test_render_ok_names(sess):
    p = sess.render(out="foo")["path"]
    assert os.path.dirname(p) == os.path.abspath(sess._render_dir()) and p.endswith("foo.png")
    assert os.path.exists(p)


def test_render_route_rejects_escape(client):
    d = client.post("/sim/render", json={"out": "/tmp/x_gz_test.png"}).json()
    assert d["ok"] is False and "invalid render filename" in d["error"]
    assert not os.path.exists("/tmp/x_gz_test.png")


def test_render_disabled_message(client, monkeypatch):
    monkeypatch.setattr(m, "RENDER_DISABLED", True)
    d = client.post("/sim/render", json={}).json()
    assert d["ok"] is False and d["error"] == "render is disabled in this environment"


# ---- 7: source is only honoured for the operator token
def test_source_forced_to_agent_without_operator_token(client, monkeypatch):
    monkeypatch.setenv("SIM_API_TOKEN", "sekrit")
    client.post("/sim/act", json={"action": {}, "source": "opponent"})
    assert client.get("/sim/state").json()["data"]["last_action"]["source"] == "agent"
    client.post("/sim/act", json={"action": {}, "source": "user"},
                headers={"Authorization": "Bearer " + m.ATK_TOKEN})
    assert client.get("/sim/state").json()["data"]["last_action"]["source"] == "agent"
    client.post("/sim/act", json={"action": {}, "source": "user"},
                headers={"Authorization": "Bearer sekrit"})
    assert client.get("/sim/state").json()["data"]["last_action"]["source"] == "user"


def test_manual_action_tagged_user(client):
    client.post("/control", json={"cmd": "manual_action", "args": {"args": {}}})
    assert client.get("/sim/state").json()["data"]["last_action"]["source"] == "user"


# ---- 8: attacker token can't fast-forward
def test_attacker_token_cannot_fast_forward(client, monkeypatch):
    monkeypatch.setenv("SIM_API_TOKEN", "sekrit")
    t0 = client.get("/sim/status").json()["data"]["t"]
    r = client.post("/sim/step", json={"n": 5}, headers={"Authorization": "Bearer " + m.ATK_TOKEN}).json()
    assert r["data"]["t"] == t0 + 1


# ---- 9: read routes gated when a token is configured
@pytest.mark.parametrize("path", ["/state", "/bench/stats", "/event"])
def test_read_routes_need_operator_token(client, monkeypatch, path):
    monkeypatch.setenv("SIM_API_TOKEN", "sekrit")
    assert client.get(path).status_code == 403
    assert client.get(path + "?token=wrong").status_code == 403
    if path != "/event":  # (SSE never ends; the 403 path is what matters)
        assert client.get(path, headers={"Authorization": "Bearer sekrit"}).status_code == 200
        assert client.get(path + "?token=sekrit").status_code == 200


# ---- 10: opponent act is invisible in the defender's last_action
def test_opponent_act_not_in_last_action(sess):
    sess.act({"redispatch": {"gen_1_0": -1.0}}, "agent")
    before = sess.latest_state()["last_action"]
    sess.act({"set_line_status": {"0_4_1": -1}}, "opponent")
    after = sess.latest_state()["last_action"]
    assert after == before and after["source"] == "agent"
    assert any(l["name"] == "0_4_1" and l["status"] == "down" for l in sess.latest_state()["lines"])


# ---- 11: stale schedule cleared on API reset
def test_reset_route_clears_attacks(client):
    ST = m.ST
    ST.sim.set_attack_schedule([{"start": 5, "end": 9, "line": "0_4_1",
                                 "action_on": {}, "action_off": {}}])
    assert ST.sim._attacks
    client.post("/sim/reset", json={})
    assert ST.sim._attacks == []


# ---- 12: trips surface in the state (what simctl observe prints)
def test_last_disc_lines_in_state(sess):
    assert sess.latest_state()["last_disc_lines"] == []
    st = None
    for _ in range(6):
        sess.step(1)
    jsonschema.validate(sess.latest_state(), SCHEMA)


# ---- 13: restart keeps the model
def test_restart_keeps_model():
    import asyncio
    from backend.app.opencode_driver import OpenCodeDriver
    d = OpenCodeDriver.__new__(OpenCodeDriver)
    seen = {}

    async def fake_create(agent="build", provider=None, model="qwen3.5-4b", variant=None):
        seen["model"], seen["variant"] = model, variant
        return None
    d.available = True
    d.session_id = None
    d.model, d.variant = "other-model", "high"
    d._emit_system = lambda *a, **k: None
    d.create_session = fake_create
    asyncio.run(d.restart())
    assert seen == {"model": "other-model", "variant": "high"}


# ---- disc_lines is a per-line cascade-level array, not a list of line ids
def test_disc_lines_names_the_lines_that_actually_tripped(sess):
    out, _, err = sess.act({"set_line_status": {"0_4_1": -1}})  # overloads 1_4_4
    assert err is None
    down_before = {l["name"] for l in sess.latest_state()["lines"] if l["status"] == "down"}
    tripped = None
    for _ in range(4):
        out, _ = sess.step(1)
        if out["disc_lines"]:
            tripped = out
            break
    assert tripped is not None, "expected 1_4_4 to trip within 3 overloaded steps"
    down_after = {l["name"] for l in sess.latest_state()["lines"] if l["status"] == "down"}
    assert set(tripped["disc_lines"]) == down_after - down_before
    assert "1_4_4" in tripped["disc_lines"]


# ---- obs.simulate() mutates the action; the real step must still see the raw action
def test_out_of_range_redispatch_is_flagged_ambiguous(sess):
    out, verb, err = sess.act({"redispatch": {"gen_1_0": -50.0}})
    assert out["ambiguous"] is True
    assert err and err.startswith("Ambiguous action")
    assert out["reward"] == 0.0
