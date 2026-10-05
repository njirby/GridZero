"""bench/score.py — the single source of truth for benchmark scoring.

L2RPN-style normalization (official 2020-2022 convention, per the benchmark plan):
  - the do-nothing (passive) agent is the reference, pinned to 0;
  - `improvement` (HEADLINE) = (S_A - S_DN)/|S_DN| on the env's own cum_reward
    (RedispReward: higher reward = lower operational cost). >0 => the agent
    operated the grid better (cheaper) than doing nothing. DoNothing => 0.
  - `norm` (L2RPN [-100,0,80,100]-style scale, secondary/readable):
        game-over episode  -> -100 * (1 - survival_frac)   (in [-100, 0], floored at 0
                              when the agent outlives the DN reference)
        completed episode  -> 80 + 20 * clamp(improvement / 0.20, 0, 1)
    Anchors: DN-not-completing = 0, completed = 80, completed + 20% improvement = 100.

Everything is computed from sim GROUND TRUTH (cumulative reward + survival),
never from anything the model is told. The model sees raw obs only.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class EpisodeResult:
    """Ground-truth outcome of ONE episode run (any agent: LLM, baseline, human)."""
    agent: str
    chronic: int                 # time serie id
    horizon: int                 # max step (the env was pinned to this)
    survived: int                # steps completed before game-over or horizon
    done: bool                   # reached the horizon (no game-over)
    cum_reward: float            # env sum of step rewards (excludes t=0 reset reward)
    ep: str = ""                 # backend episode id -> runs/<ep>.jsonl trace
    n_trips: int = 0             # protection disconnections observed
    n_illegal: int = 0           # rejected (illegal) actions
    n_ambiguous: int = 0         # rejected (ambiguous) actions
    peak_rho: float = 0.0        # max line loading fraction reached
    n_down_final: int = 0        # lines down at end
    game_over: bool = False
    agent_failed: bool = False   # infra failure: agent never operated -> EXCLUDE from scoring
    # cost / latency (LLM runs fill these; baselines leave null)
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    wall_clock_s: Optional[float] = None
    llm_turns: Optional[int] = None
    simctl_acts: Optional[int] = None
    simctl_observes: Optional[int] = None
    simctl_renders: Optional[int] = None
    error_bucket: Optional[str] = None   # filled by the error-taxonomy pass
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def survival_fraction(res: EpisodeResult, dn: EpisodeResult) -> float:
    """How far the agent got relative to do-nothing on the same chronic/horizon.

    1.0 = survived at least as long as DN (or both hit the horizon); 0 = died
    instantly. This is the reference for the game-over penalty.
    """
    if dn.survived <= 0:
        return 1.0 if res.done else 0.0
    return min(1.0, res.survived / dn.survived)


def improvement(res: EpisodeResult, dn: EpisodeResult) -> float:
    """(S_A - S_DN)/|S_DN|. The DN-pinned headline: DoNothing => 0.

    >0 means the agent operated the grid cheaper (higher reward) than passive.
    """
    if dn.cum_reward == 0:
        return 0.0
    return (res.cum_reward - dn.cum_reward) / abs(dn.cum_reward)


def normalize(res: EpisodeResult, dn: EpisodeResult) -> dict:
    """Produce the normalized score for one episode vs its DN anchor.

    Returns {norm, improvement, survival_frac}.
    """
    imp = improvement(res, dn)
    if res.done:
        norm = 80.0 + 20.0 * max(0.0, min(1.0, imp / 0.20))
    else:
        norm = -100.0 * (1.0 - survival_fraction(res, dn))
    return {"norm": norm, "improvement": imp, "survival_frac": survival_fraction(res, dn)}


def self_check():
    """Property tests for the metric (no grid2op needed)."""
    # DN that completes the horizon (e.g. a short smoke chronic): norm 80, improvement 0.
    dn = EpisodeResult(agent="dn", chronic=0, horizon=288, survived=288, done=True, cum_reward=18000.0)
    assert normalize(dn, dn)["norm"] == 80.0
    assert abs(normalize(dn, dn)["improvement"]) < 1e-12

    # Agent 20% better than a completing DN -> capped at 100.
    better = EpisodeResult(agent="x", chronic=0, horizon=288, survived=288, done=True, cum_reward=21600.0)  # +20%
    nb = normalize(better, dn)
    assert abs(nb["improvement"] - 0.20) < 1e-9
    assert nb["norm"] == 100.0, nb
    # More than 20% better also caps at 100.
    nb2 = normalize(EpisodeResult(agent="y", chronic=0, horizon=288, survived=288, done=True, cum_reward=27000.0), dn)
    assert nb2["norm"] == 100.0

    # Surviving but ~same cost as DN -> 80.
    nw = normalize(EpisodeResult(agent="w", chronic=0, horizon=288, survived=288, done=True, cum_reward=18100.0), dn)
    assert 80.0 <= nw["norm"] < 81.0, nw

    # Surviving but WORSE than DN (still completed) -> floored at 80, improvement < 0.
    nworse = normalize(EpisodeResult(agent="v", chronic=0, horizon=288, survived=288, done=True, cum_reward=15000.0), dn)
    assert nworse["norm"] == 80.0 and nworse["improvement"] < 0

    # Game over: died partway on a chronic DN completes -> between -100 and 0.
    early = EpisodeResult(agent="z", chronic=0, horizon=288, survived=100, done=False, cum_reward=800.0, game_over=True)
    nz = normalize(early, dn)
    assert -100.0 < nz["norm"] < 0.0, nz
    assert abs(nz["norm"] - (-100.0 * (1 - 100 / 288))) < 1e-9

    # Instant death -> -100.
    dead = EpisodeResult(agent="dd", chronic=0, horizon=288, survived=0, done=False, cum_reward=-10.0, game_over=True)
    assert normalize(dead, dn)["norm"] == -100.0

    # Chronic where DN ITSELF dies at 300: DN -> 0 (its own anchor); an agent dying
    # at the same step -> 0; an agent that OUTLIVES DN but still dies -> floored at 0;
    # an agent that completes -> 80+.
    dn2 = EpisodeResult(agent="dn", chronic=5, horizon=2016, survived=300, done=False, cum_reward=24000.0, game_over=True)
    assert normalize(dn2, dn2)["norm"] == 0.0
    same = EpisodeResult(agent="x", chronic=5, horizon=2016, survived=300, done=False, cum_reward=24000.0, game_over=True)
    assert normalize(same, dn2)["norm"] == 0.0
    outlive = EpisodeResult(agent="y", chronic=5, horizon=2016, survived=1200, done=False, cum_reward=90000.0, game_over=True)
    assert normalize(outlive, dn2)["norm"] == 0.0  # capped: outlived DN, floor at 0
    finish = EpisodeResult(agent="f", chronic=5, horizon=2016, survived=2016, done=True, cum_reward=120000.0)
    assert normalize(finish, dn2)["norm"] >= 80.0  # completed -> 80+
    print("score.py self_check OK")


if __name__ == "__main__":
    self_check()
