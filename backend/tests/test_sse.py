"""SSE / event-bus tests.

The /event endpoint is an infinite SSE stream; starlette's TestClient hangs
reading it (portal + streaming interaction). The real SSE-over-HTTP path is
validated by booting uvicorn (see scripts + make backend). Here we test the
EventBus that drives the stream — deterministic and fast: seq monotonicity,
STATE(latest-wins) vs LOG(append) split, and resume-after-seq replay.
"""
import os, sys, time, threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from backend.app.event_bus import EventBus  # noqa: E402
from backend.app.main import app  # noqa: E402


def test_seq_monotonic_and_state_log_split():
    bus = EventBus(ep_id="ep-test")
    for i in range(5):
        bus.emit("sim.step_outcome", {"t": i})      # LOG
    bus.emit("sim.state", {"t": 4, "n_line": 20})    # STATE
    bus.emit("sim.state", {"t": 5, "n_line": 20})    # STATE (replaces)
    s = bus.snapshot()
    states = [f for f in s if f["type"] == "sim.state"]
    # STATE is latest-wins: only ONE sim.state frame, the newest
    assert len(states) == 1
    assert states[0]["data"]["t"] == 5
    # LOG ring has the 5 step_outcomes (not the 2 states)
    states_out, log = bus.resume_after(0)
    assert sum(1 for f in log if f["type"] == "sim.step_outcome") == 5
    seqs = [f["seq"] for f in (states + log)]
    assert len(seqs) == len(set(seqs)), "seq must be unique"


def test_resume_after_seq_replays_log_tail():
    bus = EventBus(ep_id="ep-resume")
    for i in range(3):
        bus.emit("sim.step_outcome", {"t": i})
    cut = bus.last_seq
    for i in range(3, 6):
        bus.emit("sim.step_outcome", {"t": i})
    states, log = bus.resume_after(cut)
    # only the 3 events after the cut are replayed
    tail = [f for f in log if f["type"] == "sim.step_outcome"]
    assert len(tail) == 3
    assert all(f["seq"] > cut for f in tail)
    assert [f["data"]["t"] for f in tail] == [3, 4, 5]


def test_subscriber_receives_log_frames():
    bus = EventBus(ep_id="ep-sub")
    import queue
    q = bus.subscribe()
    bus.emit("sim.step_outcome", {"t": 0})
    bus.emit("sim.state", {"t": 0})
    got = []
    for _ in range(2):
        try:
            got.append(q.get(timeout=1.0))
        except queue.Empty:
            break
    types = {g["type"] for g in got}
    assert "sim.step_outcome" in types


def test_event_route_registered():
    # the /event and /api/event routes exist and are GET
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/event" in paths
    assert "/api/event" in paths
