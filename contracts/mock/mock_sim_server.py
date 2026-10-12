#!/usr/bin/env python3
"""Mock SIM-API server (stdlib only — no venv, no grid2op, no fastapi).

Implements just enough of C2 (SIM-API) + the C4 /event SSE stream, driven by
the real fixtures in contracts/examples, so that:
  - the simctl CLI (WS B) can be built and tested offline, and
  - the frontend (WS D) can be built and tested offline (recorded event stream
    + state snapshots).

It is NOT a real simulation: state is selected as the nearest fixture by `t`,
and `act` patches the affected line's status in the returned state. It is
deterministic and contract-faithful (envelope, event types, seq numbering).

Run:
  python3 contracts/mock/mock_sim_server.py --port 8731
  # then:  SIM_API_URL=http://127.0.0.1:8731 simctl status
"""
import argparse, json, os, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples")
# the --changes rules are pure stdlib: reuse the backend's module, not a copy
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from backend.app import c3_changes  # noqa: E402

HIDE_HORIZON = os.environ.get("SIM_HIDE_HORIZON", "") == "1"
HISTORY_FIELDS = {"line": ("rho", "p_or", "status"), "gen": ("p", "target_dispatch", "actual_dispatch"),
                  "sub": ("p",)}


def _agent_view(d):
    if HIDE_HORIZON and isinstance(d, dict) and "max_t" in d:
        return {k: v for k, v in d.items() if k != "max_t"}
    return d


def _fill_optional(s):
    """The example fixtures predate the optional C3 fields (buses, overflow
    counter, dispatch); derive them from what the fixture does record."""
    s.setdefault("overflow_allowed", 2)
    moved = ((s.get("last_action") or {}).get("args") or {}).get("change_bus") or {}
    moved_or, moved_ex = set(moved.get("lines_or_id") or []), set(moved.get("lines_ex_id") or [])
    for l in s["lines"]:
        up = l["status"] != "down"
        l.setdefault("or_bus", (2 if l["name"] in moved_or else 1) if up else -1)
        l.setdefault("ex_bus", (2 if l["name"] in moved_ex else 1) if up else -1)
        l.setdefault("timestep_overflow", 1 if l.get("overflow") else 0)
    for g in s.get("gens", []):
        g.setdefault("connected", True)
        g.setdefault("target_dispatch", 0.0)
        g.setdefault("actual_dispatch", 0.0)
    return s

def load(fn):
    with open(os.path.join(EX, fn)) as f:
        return json.load(f)

META = load("grid-meta.json")
S_T0 = load("grid-state-t0.json")
S_T50 = load("grid-state-t50.json")
S_T120 = load("grid-state-t120.json")
FIXTURES = [(0, S_T0), (50, S_T50), (120, S_T120)]
MAX_T = 8064


class MockSim:
    def __init__(self):
        self.t = 0
        self.lock = threading.RLock()
        self.seq = 469
        self.view = None  # --changes baseline (single viewer)
        self.last_reward = S_T0["reward"]

    def next_seq(self):
        self.seq += 1
        return self.seq

    def nearest(self, t):
        return min(FIXTURES, key=lambda ft: abs(ft[0] - t))  # (t, state)

    def state(self):
        with self.lock:
            _, s = self.nearest(self.t)
            s = json.loads(json.dumps(s))  # deep copy
            s["t"] = self.t
            s["png"] = f"render/t{self.t:04d}.png"
            return _fill_optional(s)

    def reset(self):
        with self.lock:
            self.t = 0
            self.view = None
            return self.state()

    def view_full(self):
        with self.lock:
            s = self.state()
            self.view = c3_changes.new_view(s)
            return s

    def history(self, kind, name, last):
        """Readings of one element for the last `last` steps, taken from the
        nearest fixture at each t (the mock has no real dynamics)."""
        fields = HISTORY_FIELDS.get(kind)
        if fields is None:
            raise KeyError(f"unknown kind {kind!r} (valid: {', '.join(HISTORY_FIELDS)})")
        with self.lock:
            names = [e["name"] for e in S_T0[kind + "s"]]
            idx = (names.index(name) if name in names else names.index(f"sub_{name}")
                   if kind == "sub" and f"sub_{name}" in names else None)
            if idx is None:
                raise KeyError(f"unknown {kind} {name!r}")
            last = max(1, min(int(last), 288))
            rows = []
            for t in range(max(0, self.t - last + 1), self.t + 1):
                e = _fill_optional(json.loads(json.dumps(self.nearest(t)[1])))[kind + "s"][idx]
                rows.append({"t": t, **{f: e[f] for f in fields}})
            return {"kind": kind, "name": names[idx], "fields": list(fields), "rows": rows, "max_last": 288}

    def changes(self, rho_eps, mw_eps):
        with self.lock:
            out, self.view = c3_changes.changes(self.state(), self.view, rho_eps, mw_eps)
            return out

    def step(self, n):
        with self.lock:
            self.t = min(self.t + max(1, n), MAX_T)
            s = self.state()
            return {"t": s["t"], "reward": s["reward"], "cum_reward": s["cum_reward"],
                    "done": s["t"] >= MAX_T, "lines_down": s["n_down"],
                    "overloads": [l["name"] for l in s["lines"] if l["overflow"]],
                    "disc_lines": []}, {"is_illegal": False, "is_ambiguous": False,
                    "opponent_attack_line": None}

    def act(self, action, source="agent"):
        with self.lock:
            self.t = min(self.t + 1, MAX_T)
            s = self.state()
            applied = {}
            new_overloads = []
            # apply set_line_status / change_line_status to the state
            for key in ("set_line_status", "change_line_status"):
                for line_name, val in (action.get(key) or {}).items():
                    ln = next((l for l in s["lines"] if l["name"] == line_name), None)
                    if ln is None:
                        return ({"t": s["t"], "reward": s["reward"], "cum_reward": s["cum_reward"],
                                 "done": False, "disc_lines": [], "new_overloads": [],
                                 "illegal": True, "ambiguous": False, "applied": applied},
                                {"is_illegal": True, "is_ambiguous": False},
                                f"Illegal — no line named '{line_name}'")
                    if key == "set_line_status":
                        ln["status"] = "up" if val == 1 else "down" if val == -1 else ln["status"]
                        ln["rho"] = 0.0 if ln["status"] == "down" else ln["rho"]
                        if ln["status"] == "down":
                            ln["or_bus"] = ln["ex_bus"] = -1
                        applied[key] = {line_name: "up" if val == 1 else "down"}
                    elif key == "change_line_status":
                        ln["status"] = "up" if ln["status"] == "down" else "down"
                        applied[key] = {line_name: ln["status"]}
            s["n_down"] = sum(1 for l in s["lines"] if l["status"] == "down")
            s["max_rho"] = max((l["rho"] for l in s["lines"]), default=0.0)
            s["n_overflow"] = sum(1 for l in s["lines"] if l["overflow"])
            new_overloads = [l["name"] for l in s["lines"] if l["overflow"]]
            s["last_action"] = {"source": source, "summary": json.dumps(action),
                                "args": action, "t": s["t"] - 1}
            return ({"t": s["t"], "reward": s["reward"], "cum_reward": s["cum_reward"],
                     "done": False, "disc_lines": [], "new_overloads": new_overloads,
                     "illegal": False, "ambiguous": False, "applied": applied},
                    {"is_illegal": False, "is_ambiguous": False,
                     "predicted_disc_lines": new_overloads}, None)

    def render(self, width):
        with self.lock:
            base = os.path.abspath(os.path.join(HERE, ".."))
            path = os.path.join(base, "examples", "render",
                                f"t{self.t:04d}.png")
            # fall back to a real fixture png if exact t has none
            if not os.path.exists(path):
                path = os.path.join(base, "examples", "render",
                                    f"t{self.nearest(self.t)[0]:04d}.png")
            return {"path": path, "width": width or 800, "height": 500, "t": self.t}


SIM = MockSim()


def envelope(ok, data, error=None, verbose=None):
    return {"ok": ok, "data": data, "error": error, "verbose": verbose or {}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        ln = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(ln) or b"{}") if ln else {}

    def do_GET(self):
        p = self.path.split("?")[0]
        q = parse_qs(urlparse(self.path).query)
        if p == "/sim/status":
            s = SIM.state()
            self._json(envelope(True, _agent_view({"up": True, "env": s["env"], "t": s["t"],
                                       "max_t": s["max_t"], "reward": s["reward"],
                                       "cum_reward": s["cum_reward"], "done": s["done"]})))
        elif p == "/sim/state":
            full = q.get("view", [""])[0] == "full"
            self._json(envelope(True, _agent_view(SIM.view_full() if full else SIM.state())))
        elif p == "/sim/history":
            try:
                d = SIM.history(q.get("kind", [""])[0], q.get("name", [""])[0], q.get("last", ["12"])[0])
            except KeyError as e:
                return self._json(envelope(False, None, error=e.args[0]))
            self._json(envelope(True, d))
        elif p == "/sim/changes":
            try:
                rho_eps = float(q.get("rho_eps", [os.environ.get("SIM_CHANGES_RHO_EPS") or c3_changes.DEFAULT_RHO_EPS])[0])
                mw_eps = float(q.get("mw_eps", [os.environ.get("SIM_CHANGES_MW_EPS") or c3_changes.DEFAULT_MW_EPS])[0])
                assert rho_eps >= 0 and mw_eps >= 0
            except (ValueError, AssertionError):
                return self._json(envelope(False, None, error="rho_eps/mw_eps must be >= 0"), 400)
            self._json(envelope(True, _agent_view(SIM.changes(rho_eps, mw_eps))))
        elif p == "/state":
            s = SIM.state()
            self._json({"sim": s, "mode": "agent", "running": True, "last_seq": SIM.seq})
        elif p == "/api/grid/meta":
            self._json(META)
        elif p == "/event":
            self._sse()
        elif p == "/health":
            self._json({"healthy": True, "mock": True})
        else:
            self._json(envelope(False, None, error=f"no mock route {p}"), 404)

    def do_POST(self):
        p = self.path.split("?")[0]
        b = self._body()
        if p == "/sim/reset":
            self._json(envelope(True, SIM.reset()))
        elif p == "/sim/step":
            data, verb = SIM.step(int(b.get("n", 1)))
            self._json(envelope(True, data, verbose=verb))
        elif p == "/sim/act":
            data, verb, err = SIM.act(b.get("action") or {}, source=b.get("source", "agent"))
            if err is not None:
                self._json(envelope(False, data, error=err, verbose=verb), 200)
            else:
                self._json(envelope(True, data, verbose=verb))
        elif p == "/sim/attack":
            # adversarial action: same path as act but tagged 'opponent'
            data, verb, err = SIM.act(b.get("action") or {}, source="opponent")
            if err is not None:
                self._json(envelope(False, data, error=err, verbose=verb), 200)
            else:
                self._json(envelope(True, data, verbose=verb))
        elif p == "/sim/render":
            self._json(envelope(True, SIM.render(int(b.get("width", 0) or 0))))
        elif p == "/control":
            self._json(envelope(True, {"cmd": b.get("cmd"), "ack": True}))
        else:
            self._json(envelope(False, None, error=f"no mock route {p}"), 404)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        nd = os.path.join(EX, "event-stream.ndjson")
        c3 = {0: S_T0, 50: S_T50, 120: S_T120}
        for line in open(nd):
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            d = ev.get("data")
            if isinstance(d, str) and d.startswith("<<C3"):
                key = d.split("grid-state-")[1]
                ev["data"] = c3[int("".join(ch for ch in key if ch.isdigit()))]
            self.wfile.write(f"id: {ev['seq']}\n".encode())
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.write(b"data: {\"seq\":0,\"type\":\"ping\",\"ts\":"
                         + str(time.time()).encode() + b",\"data\":{}}\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"mock SIM-API on http://{a.host}:{a.port}  (fixtures: {EX})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
