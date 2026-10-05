"""event_bus.py — global seq counter, ring buffer, subscribers, STATE/LOG, JSONL.

Two event classes (see contracts/C4):
  STATE — latest-wins, replaceable, coalesced to <=10Hz per subscriber.
  LOG   — append-only, replayable, never dropped within the ring.

The bus assigns `seq`, persists LOG to runs/<ep>.jsonl, and fans frames out to
subscribers. Subscribers are thread-safe queue.Queue objects; the SSE handler
drains its queue via asyncio.to_thread (the sim runs in worker threads, so we
must not use an asyncio.Queue from multiple threads).
"""
from __future__ import annotations
import json, os, queue, threading, time
from collections import deque

STATE_TYPES = {"sim.state", "session.status"}
LOG_TYPES = {"sim.step_outcome", "agent.delta", "agent.tool_call", "agent.tool_result",
             "agent.turn_end", "user.action", "opponent.action", "opponent.step",
             "system", "episode.summary",
             # agent-vs-agent: the ATTACKER's opencode session, tagged separately so
             # its events never mix into the defender's agent.* trace (blinding).
             "attacker.delta", "attacker.tool_call", "attacker.tool_result",
             "attacker.turn_end"}


class EventBus:
    def __init__(self, runs_dir="runs", ep_id="ep-local", ring=5000, state_hz=10):
        self._seq = 0
        self._lock = threading.Lock()
        self._ring = deque(maxlen=ring)
        self._latest_state = {}
        self._subs = []               # list[queue.Queue]
        self._ep = ep_id
        self._runs_dir = runs_dir
        self._log_file = None
        self._state_min_interval = 1.0 / state_hz
        self._last_state_ts = {}      # id(queue) -> ts

    # ---- lifecycle ----
    def start_episode(self, ep_id, runs_dir=None):
        with self._lock:
            self._ep = ep_id
            if runs_dir:
                self._runs_dir = runs_dir
            if self._log_file:
                try:
                    self._log_file.close()
                except Exception:
                    pass
            try:
                os.makedirs(self._runs_dir, exist_ok=True)
                self._log_file = open(os.path.join(self._runs_dir, f"{ep_id}.jsonl"), "a")
            except Exception:
                self._log_file = None

    def stop(self):
        with self._lock:
            if self._log_file:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None

    # ---- subscribe (thread-safe) ----
    def subscribe(self) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=4000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs = [s for s in self._subs if s is not q]

    @property
    def last_seq(self):
        with self._lock:
            return self._seq

    @property
    def episode(self):
        return self._ep

    # ---- emit (thread-safe; callable from any thread) ----
    def emit(self, etype, data):
        if etype not in STATE_TYPES and etype not in LOG_TYPES:
            return
        is_state = etype in STATE_TYPES
        now = time.time()
        with self._lock:
            self._seq += 1
            frame = {"seq": self._seq, "type": etype, "ts": now, "data": data}
            if is_state:
                self._latest_state[etype] = frame
            else:
                self._ring.append(frame)
                if self._log_file:
                    try:
                        self._log_file.write(json.dumps(frame) + "\n")
                        self._log_file.flush()
                    except Exception:
                        pass
            for q in self._subs:
                if is_state:
                    if now - self._last_state_ts.get(id(q), 0.0) < self._state_min_interval:
                        continue
                    self._last_state_ts[id(q)] = now
                self._safe_put(q, frame, is_state)

    def _safe_put(self, q, frame, is_state):
        try:
            if q.full():
                if is_state:
                    return
                try:
                    q.get_nowait()
                except Exception:
                    pass
            q.put_nowait(frame)
        except Exception:
            pass

    # ---- resume ----
    def snapshot(self):
        with self._lock:
            return list(self._latest_state.values())

    def resume_after(self, after_seq):
        with self._lock:
            states = list(self._latest_state.values())
            log = [f for f in self._ring if f["seq"] > after_seq]
        return states, log
