import os


def test_sim_state_envelope(client):
    d = client.get("/sim/state").json()
    assert d["ok"] is True
    assert d["data"]["n_line"] == 20
    assert "lines" in d["data"] and len(d["data"]["lines"]) == 20


def test_sim_status(client):
    d = client.get("/sim/status").json()
    assert d["ok"] is True and d["data"]["up"] is True


def test_sim_step(client):
    r = client.post("/sim/step", json={"n": 2})
    d = r.json()
    assert d["ok"] is True and d["data"]["t"] >= 2


def test_sim_act_ok(client):
    r = client.post("/sim/act", json={"action": {}})  # neutral no-op
    d = r.json()
    assert d["ok"] is True and d["data"]["illegal"] is False


def test_sim_act_illegal(client):
    r = client.post("/sim/act", json={"action": {"set_line_status": {"99_99_9": -1}}})
    d = r.json()
    assert d["ok"] is False and d["error"]
    assert d["verbose"]["is_illegal"] is True


def test_sim_act_malformed(client):
    r = client.post("/sim/act", json={"action": "not-a-dict"})
    assert r.status_code == 400


def test_sim_render_writes_png(client):
    d = client.post("/sim/render", json={"width": 800}).json()
    assert d["ok"] is True
    assert os.path.exists(d["data"]["path"]) and os.path.getsize(d["data"]["path"]) > 0


def test_state_matches_sim_state(client):
    a = client.get("/state").json()["sim"]
    b = client.get("/sim/state").json()["data"]
    assert a["t"] == b["t"] and a["n_line"] == b["n_line"]


def test_meta(client):
    m = client.get("/api/grid/meta").json()
    assert m["n_sub"] == 14 and len(m["subs"]) == 14


def test_attack_opponent(client):
    # adversarial action: opens a line, tagged 'opponent'; the line flips and the
    # state's last_action records source=opponent (the defender is blind to who).
    r = client.post("/sim/attack", json={"action": {"set_line_status": {"0_4_1": -1}}})
    d = r.json()
    assert d["ok"] is True and d["data"]["applied"] == {"set_line_status": {"0_4_1": "down"}}
    s = client.get("/sim/state").json()["data"]
    line = [l for l in s["lines"] if l["name"] == "0_4_1"][0]
    assert line["status"] == "down"
    assert s["last_action"]["source"] == "opponent"


def test_token_guard_reset_control(client, monkeypatch):
    monkeypatch.setenv("SIM_API_TOKEN", "sekrit")
    h = {"Authorization": "Bearer sekrit"}
    assert client.post("/sim/reset", json={}).status_code == 403
    assert client.post("/control", json={"cmd": "pause"}).status_code == 403
    assert client.post("/sim/reset", json={}, headers=h).status_code == 200
    assert client.post("/control", json={"cmd": "release"}, headers=h).status_code == 200


def test_token_guard_step_clamp(client, monkeypatch):
    monkeypatch.setenv("SIM_API_TOKEN", "sekrit")
    t0 = client.get("/sim/status").json()["data"]["t"]
    r = client.post("/sim/step", json={"n": 5}).json()
    assert r["ok"] is True and r["data"]["t"] == t0 + 1  # clamped to 1
    r2 = client.post("/sim/step", json={"n": 5},
                     headers={"Authorization": "Bearer sekrit"}).json()
    assert r2["ok"] is True and r2["data"]["t"] == t0 + 6


def test_token_guard_attack(client, monkeypatch):
    import backend.app.main as m
    monkeypatch.setenv("SIM_API_TOKEN", "sekrit")
    spec = {"set_line_status": {"0_4_1": -1}}
    assert client.post("/sim/attack", json=spec).status_code == 403
    assert client.post("/sim/attack", json=spec,
                       headers={"Authorization": "Bearer " + m.ATK_TOKEN}).status_code == 200
    # operator token also works
    assert client.post("/sim/attack", json=spec,
                       headers={"Authorization": "Bearer sekrit"}).status_code == 200
