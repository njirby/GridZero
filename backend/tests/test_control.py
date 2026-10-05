import json


def test_control_pause_resume(client):
    client.post("/control", json={"cmd": "pause"})
    assert client.get("/state").json()["mode"] == "paused"
    client.post("/control", json={"cmd": "resume"})
    assert client.get("/state").json()["mode"] == "agent"


def test_control_take_over_release(client):
    client.post("/control", json={"cmd": "take_over"})
    assert client.get("/state").json()["mode"] == "manual"
    client.post("/control", json={"cmd": "release"})
    assert client.get("/state").json()["mode"] == "agent"


def test_control_single_step(client):
    before = client.get("/sim/state").json()["data"]["t"]
    client.post("/control", json={"cmd": "single_step"})
    after = client.get("/sim/state").json()["data"]["t"]
    assert after == before + 1


def test_control_instruction_does_not_crash(client):
    r = client.post("/control", json={"cmd": "instruction", "args": {"text": "reduce losses"}})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_user_action_and_system_logged(client):
    # control returns ok; the /state endpoint reflects the mode change (the
    # user.action + system event emission is covered at the EventBus level in
    # test_sse.py — TestClient can't read the infinite SSE stream without hanging).
    r = client.post("/control", json={"cmd": "pause"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get("/state").json()["mode"] == "paused"
