"""sim_session.py — the single-writer owner of the grid2op env.

All env.reset/step/act calls happen under self._lock, so the sim has exactly
one logical writer (the contract's "single writer" guarantee) even though the
FastAPI async routes call it via asyncio.to_thread. A SimBackend interface is
kept so the env could later move to a subprocess without redesigning callers.

Reward semantics (C3): `reward` = most recent step's grid2op reward (the reset
value at t=0); `cum_reward` = sum of COMPLETED step rewards only (0 at t=0, the
t=0 reset reward is excluded).
"""
from __future__ import annotations
import os, threading
import numpy as np

from .c3 import to_c3, build_meta


class SimSession:
    def __init__(self, env_name="l2rpn_case14_sandbox", env=None, root=None):
        self.env = env_name
        self.root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._lock = threading.RLock()
        self._cum_reward = 0.0
        self._last_reward = None
        self._last_action = None
        self._obs = None
        self._last_info = None
        self._env = env  # injectable for tests
        self._attached = False
        # long-horizon robustness: cache the last good C3 + finished flag so we
        # never touch a terminal grid2op env (which requires reset() to re-access).
        self._last_c3 = None
        self._finished = False
        self._chronic = None
        self._horizon = 0
        # adversarial mode: a deterministic attack schedule fired by the backend at
        # exact sim-steps (fair regardless of defender pace). Each attack is a dict
        # {start, end, line, action_on, action_off}. Defender is blind to it.
        self._attacks = []
        self._atk_started = set()
        self._atk_ended = set()
        self._in_attack = False
        self._pending_opponent = []   # fired-attack events for the backend to emit

    # ---- env lifecycle ----
    def _ensure_env(self):
        if self._env is None:
            import warnings
            warnings.filterwarnings("ignore")
            import grid2op
            self._env = grid2op.make(self.env)
        return self._env

    def _ensure_renderer(self):
        env = self._ensure_env()
        if not self._attached:
            env.attach_renderer()
            self._attached = True
        return env

    def _render_dir(self):
        d = os.path.join(self.root, "render")
        os.makedirs(d, exist_ok=True)
        return d

    # ---- SimBackend interface ----
    def metadata(self) -> dict:
        with self._lock:
            return build_meta(self._ensure_env())

    def status(self) -> dict:
        with self._lock:
            c3 = self._last_c3
            if c3 is None:
                return {"up": self._env is not None, "env": self.env, "t": 0,
                        "max_t": 0, "reward": 0.0, "cum_reward": 0.0, "done": False}
            return {"up": True, "env": self.env,
                    "t": c3["t"], "max_t": c3["max_t"],
                    "reward": c3["reward"], "cum_reward": c3["cum_reward"],
                    "done": self._finished}

    def latest_state(self) -> dict:
        with self._lock:
            return self._last_c3 or {}

    def _png(self):
        la = self._last_action
        return la.get("png", "") if la else ""

    # ---- operations ----
    def reset(self, env_name=None, seed=None, options=None) -> dict:
        """New episode. Benchmark knobs (verified on 1.12.5):
        options={"max step": N}  -> horizon; options={"time serie id": k} -> pin chronic.
        seed -> RNG seed (determinism for stochastic agents)."""
        with self._lock:
            if env_name:
                self.env = env_name
            self._ensure_env()
            self._ensure_renderer()
            obs = self._env.reset(seed=seed, options=options)
            self._obs = obs
            self._last_info = None
            self._cum_reward = 0.0
            self._last_reward = float(self._env.current_reward)
            self._last_action = None
            self._chronic = (options or {}).get("time serie id")
            self._horizon = int(obs.max_step)
            # benchmark safety counters
            self._n_trips = 0
            self._n_illegal = 0
            self._n_ambiguous = 0
            self._peak_rho = 0.0
            self._game_over = False
            self._finished = False
            # fresh fire-tracking for this episode (re-fires the schedule from t=0)
            self._atk_started = set()
            self._atk_ended = set()
            self._pending_opponent = []
            self._last_c3 = to_c3(self._env, obs, self._last_reward, self._cum_reward, None, "")
            return self._last_c3

    def _after_step(self, obs, reward, done, info):
        self._obs = obs
        self._last_info = info
        self._cum_reward += float(reward)
        self._last_reward = float(reward)
        # benchmark safety counters
        try:
            disc = np.asarray(info.get("disc_lines", []), dtype=int)
            self._n_trips += int(sum(1 for j in disc if j >= 0))
        except Exception:
            pass
        self._n_illegal += int(bool(info.get("is_illegal")))
        self._n_ambiguous += int(bool(info.get("is_ambiguous")))
        rho = np.asarray(obs.rho, dtype=float)
        if rho.size:
            self._peak_rho = max(self._peak_rho, float(np.nanmax(rho)))
        t = int(obs.current_step)
        if done and t < self._horizon:
            self._game_over = True
        self._finished = self._game_over or t >= self._horizon
        # refresh the cached C3 while the env is still accessible
        self._last_c3 = to_c3(self._env, obs, self._last_reward, self._cum_reward,
                              self._last_action, f"t{t:04d}.png")
        # adversarial: fire any scheduled attacks that crossed a step boundary
        self._fire_attacks()

    def episode_stats(self) -> dict:
        with self._lock:
            c3 = self._last_c3 or {}
            t = c3.get("t", 0)
            n_down = int(sum(1 for l in c3.get("lines", []) if l["status"] != "up"))
            return {"chronic": self._chronic, "horizon": self._horizon, "t": t,
                    "survived": t, "done": (not self._game_over) and t >= self._horizon,
                    "game_over": self._game_over, "finished": self._finished,
                    "cum_reward": float(self._cum_reward),
                    "n_trips": self._n_trips, "n_illegal": self._n_illegal,
                    "n_ambiguous": self._n_ambiguous, "peak_rho": round(self._peak_rho, 4),
                    "n_down_final": n_down}

    def _finished_outcome(self):
        c3 = self._last_c3 or {}
        return ({"t": c3.get("t", 0), "reward": c3.get("reward", 0.0),
                 "cum_reward": c3.get("cum_reward", 0.0), "done": True,
                 "disc_lines": [], "new_overloads": [], "illegal": False,
                 "ambiguous": False, "applied": {}},
                {"is_illegal": False, "is_ambiguous": False, "opponent_attack_line": None})

    def set_attack_schedule(self, attacks):
        """Install a deterministic attack schedule (list of dicts: start, end, line,
        action_on, action_off). Reset the fire-tracking. Called before reset()."""
        with self._lock:
            self._attacks = sorted(attacks, key=lambda a: a["start"])
            self._atk_started = set()
            self._atk_ended = set()

    def _opponent_step(self, action_dict):
        """Apply one adversarial action as an env step (advances t, real dynamics).
        Updates the benchmark counters + C3 cache; does NOT recurse into _after_step."""
        A = self._env.action_space
        try:
            act = A(action_dict)
        except Exception:
            act = A()  # if the attack action is illegal at this instant, no-op the step
        obs, reward, done, info = self._env.step(act)
        self._obs = obs
        self._last_info = info
        self._cum_reward += float(reward)
        self._last_reward = float(reward)
        try:
            disc = np.asarray(info.get("disc_lines", []), dtype=int)
            self._n_trips += int(sum(1 for j in disc if j >= 0))
        except Exception:
            pass
        self._n_illegal += int(bool(info.get("is_illegal")))
        rho = np.asarray(obs.rho, dtype=float)
        if rho.size:
            self._peak_rho = max(self._peak_rho, float(np.nanmax(rho)))
        t = int(obs.current_step)
        if done and t < self._horizon:
            self._game_over = True
        self._finished = self._game_over or t >= self._horizon
        self._last_c3 = to_c3(self._env, obs, self._last_reward, self._cum_reward,
                              self._last_action, f"t{t:04d}.png")
        # caller (backend) emits the opponent.step event; we return the t for that
        return t

    def _fire_attacks(self):
        """Fire any scheduled attacks whose boundary <= current t. Called from
        _after_step (i.e. at every sim step, so it's pace-independent). Records
        each fired attack in _pending_opponent for the backend to emit."""
        if not self._attacks or self._in_attack or self._finished:
            return
        t0 = int(self._obs.current_step)
        self._in_attack = True
        try:
            for idx, atk in enumerate(self._attacks):
                if idx not in self._atk_started and atk["start"] <= t0 and atk["start"] > 0:
                    self._atk_started.add(idx)
                    self._opponent_step(atk["action_on"])
                    self._pending_opponent.append({"t": atk["start"], "kind": "start",
                                                   "line": atk["line"], "action": atk["action_on"]})
                elif idx in self._atk_started and idx not in self._atk_ended and atk["end"] <= int(self._obs.current_step):
                    self._atk_ended.add(idx)
                    self._opponent_step(atk["action_off"])
                    self._pending_opponent.append({"t": atk["end"], "kind": "end",
                                                   "line": atk["line"], "action": atk["action_off"]})
        finally:
            self._in_attack = False

    def drain_opponent_events(self):
        """Pop and return pending fired-attack events (backend emits them)."""
        with self._lock:
            out, self._pending_opponent = self._pending_opponent, []
            return out

    def step(self, n=1) -> tuple:
        with self._lock:
            if self._finished:
                out, verb = self._finished_outcome()
                return out, verb
            self._ensure_env()
            A = self._env.action_space
            n = max(1, min(int(n), 50))
            prev_rho = np.asarray(self._obs.rho, dtype=float) if self._obs is not None else None
            self._last_info = None
            for _ in range(n):
                if self._finished:
                    break
                obs, reward, done, info = self._env.step(A())
                self._after_step(obs, reward, done, info)
            return self._outcome_after(prev_rho, source="auto", applied=None)

    def act(self, action_dict: dict, source="agent") -> tuple:
        """source: 'agent' (the LLM/human via simctl) or 'opponent' (adversarial
        attack). Opponent actions step the SAME single-writer env but are tagged
        'opponent' in the trace; the defender is never told WHO acted — it only sees
        the effect in its next observe. This is the adversarial blindness."""
        with self._lock:
            if self._finished:
                out, verb = self._finished_outcome()
                return out, verb, ("Episode is over (reached horizon or game-over); "
                                   "run `simctl reset` to start a new one.")
            self._ensure_env()
            A = self._env.action_space
            try:
                act = A(action_dict)
            except Exception as e:
                cur = self.latest_state()
                outcome = {"t": cur.get("t", 0), "reward": cur.get("reward", 0.0),
                           "cum_reward": cur.get("cum_reward", 0.0), "done": self._finished,
                           "disc_lines": [], "new_overloads": [],
                           "illegal": True, "ambiguous": False, "applied": {}}
                verbose = {"is_illegal": True, "is_ambiguous": True, "predicted_disc_lines": []}
                return outcome, verbose, f"Illegal — {e}"
            # dry-run to predict trips (only meaningful for agent actions; skip for opponent)
            predicted = []
            if source != "opponent":
                try:
                    _do, _dr, _dd, d_info = self._obs.simulate(act)
                    disc = np.asarray(d_info.get("disc_lines", []), dtype=int)
                    predicted = [str(self._env.name_line[j]) for j in disc if j is not None and j >= 0]
                except Exception:
                    pass
            prev_rho = np.asarray(self._obs.rho, dtype=float)
            applied = describe_applied(action_dict)
            self._last_action = {"source": source, "summary": summarize_applied(applied),
                                 "args": action_dict, "t": int(self._obs.current_step)}
            obs, reward, done, info = self._env.step(act)
            self._after_step(obs, reward, done, info)
            out, verb = self._outcome_after(prev_rho, source=source, applied=applied)
            verb["predicted_disc_lines"] = predicted
            err = None
            if info.get("is_illegal") or info.get("is_ambiguous"):
                err = (info.get("reason_alarm_illegal") or info.get("reason_alert_illegal")
                       or ("Illegal action" if info.get("is_illegal") else "Ambiguous action"))
            return out, verb, err

    def _outcome_after(self, prev_rho, source, applied) -> tuple:
        env = self._env
        obs = self._obs
        cur_rho = np.asarray(obs.rho, dtype=float)
        new_overloads = [str(env.name_line[i]) for i in range(env.n_line)
                         if cur_rho[i] > 1.0 and (prev_rho is None or prev_rho[i] <= 1.0)]
        disc_lines = []
        if self._last_info is not None:
            try:
                disc = np.asarray(self._last_info.get("disc_lines", []), dtype=int)
                disc_lines = [str(env.name_line[j]) for j in disc if j is not None and j >= 0]
            except Exception:
                disc_lines = []
        cur = self.latest_state()
        outcome = {"t": int(obs.current_step), "reward": round(self._last_reward, 4),
                   "cum_reward": round(self._cum_reward, 4), "done": self._finished,
                   "disc_lines": disc_lines, "new_overloads": new_overloads,
                   "illegal": bool(self._last_info.get("is_illegal")) if self._last_info else False,
                   "ambiguous": bool(self._last_info.get("is_ambiguous")) if self._last_info else False,
                   "applied": applied or {}}
        verbose = {"is_illegal": outcome["illegal"], "is_ambiguous": outcome["ambiguous"],
                   "opponent_attack_line": None}
        if self._last_info:
            verbose["opponent_attack_line"] = self._last_info.get("opponent_attack_line")
        return outcome, verbose

    def render(self, width=800, out=None) -> dict:
        with self._lock:
            self._ensure_renderer()
            d = self._render_dir()
            t = int(self._obs.current_step) if self._obs is not None else 0
            fname = (out or f"t{t:04d}")
            if not fname.endswith(".png"):
                fname += ".png"
            path = os.path.join(d, fname)
            self._env.render()
            self._env.viewer_fig.savefig(path, format="png", dpi=90, bbox_inches="tight")
            w, h = width, 500
            return {"path": os.path.abspath(path), "width": w, "height": h, "t": t}


def describe_applied(action_dict):
    out = {}
    for k, v in (action_dict or {}).items():
        if isinstance(v, dict):
            out[k] = {}
            for name, val in v.items():
                if k == "set_line_status":
                    out[k][str(name)] = "up" if val == 1 else "down" if val == -1 else "noop"
                else:
                    out[k][str(name)] = val
        else:
            out[k] = v
    return out


def summarize_applied(applied):
    bits = []
    for k, v in applied.items():
        if isinstance(v, dict):
            bits += [f"{k} {name}={val}" for name, val in v.items()]
        else:
            bits.append(f"{k}={v}")
    return " ".join(bits)
