"""bench/agents.py — lightweight rule-based expert baseline for the benchmark.

grid2op ships DoNothingAgent / RandomAgent (fast, exact) and an "expert" family.
That family is the wrong shape for this benchmark: RecoPowerline/AlertAgent only
*reconnect* tripped lines (a strict tie with do-nothing on the sandbox, which has
no opponent to recover from), and TopologyGreedy is too slow to run per-episode
(~14 s/step -> ~30 h for one 8064-step episode: it simulates the whole space).

N1GreedyAgent is the *preventive* expert: it stays hands-off while the grid is
healthy, and acts only when a line is PREDICTED to trip within the next 2 steps
(the env's own overflow counter). Then it simulates a handful of targeted N-1
candidates (open the line's parallel paths / re-route a busbar / redispatch the
host generator) over a 2-step rollout and commits the one that strictly beats
do-nothing. ~5 simulates/step -> tens of ms/step -> a full 8064-step episode in
~30 min. Same `act(obs, reward, done) -> Action` interface as the shipped agents,
and it uses only `obs.simulate` (the exact tool the LLM's docs recommend), so it
is a fair "smart non-LLM" bar.
"""
from __future__ import annotations
import numpy as np
from grid2op.Agent import BaseAgent


class N1GreedyAgent(BaseAgent):
    def __init__(self, action_space, lookahead=2, n_candidates=6):
        super().__init__(action_space)
        self._la = lookahead
        self._n = n_candidates

    # --- rollout helpers (pure simulation, never mutates the env) ---
    def _rollout_max_trip(self, obs, action, horizon):
        """max overflow counter reached over `horizon` sim-steps; None on error."""
        o = obs
        try:
            worst = 0
            for _ in range(horizon):
                o, _, done, _ = o.simulate(action, time_step=1)
                if done:
                    return 10**6  # game over during rollout = worst
                worst = max(worst, int(np.max(np.asarray(o.timestep_overflow, dtype=int))))
            return worst
        except Exception:
            return None

    def _gen_at(self, obs, sub_name):
        for j in range(obs.n_gen):
            if str(obs.name_sub[int(obs.gen_to_subid[j])]) == sub_name and bool(obs.gen_redispatchable[j]):
                return str(obs.name_gen[j])
        return None

    def act(self, observation, reward, done=False):
        obs = observation
        A = self.action_space
        overflow = np.asarray(obs.timestep_overflow, dtype=int)
        worst_i = int(np.argmax(overflow))
        # only act when a trip is imminent (overflow counter at the allowed max
        # means the next overload step trips) or the grid is already over 1.0
        rho = np.asarray(obs.rho, dtype=float)
        imminent = (overflow[worst_i] >= 1) or (rho[worst_i] > 1.0)
        if not imminent:
            return A()

        name = str(obs.name_line[worst_i])
        osub = str(obs.name_sub[int(obs.line_or_to_subid[worst_i])])
        esub = str(obs.name_sub[int(obs.line_ex_to_subid[worst_i])])
        cands = []
        cands.append({"set_line_status": {name: -1}})                       # open it
        cands.append({"change_bus": {"lines_or_id": [name]}})               # re-route or end
        cands.append({"change_bus": {"lines_ex_id": [name]}})               # re-route ex end
        for sub in (osub, esub):
            g = self._gen_at(obs, sub)
            if g:
                cands.append({"redispatch": {g: -5.0}})                     # shed host gen
                break
        cands = cands[: self._n]

        base = self._rollout_max_trip(obs, A(), self._la)   # do-nothing outcome
        best_a, best_score = A(), base
        for c in cands:
            s = self._rollout_max_trip(obs, A(c), self._la)
            if s is None:
                continue
            if base is None or s < best_score:
                best_a, best_score = A(c), s
        # commit only if strictly better than doing nothing (else don't churn)
        if base is not None and best_score >= base:
            return A()
        return best_a

    def seed(self, _seed=None):
        pass
