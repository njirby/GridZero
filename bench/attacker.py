"""bench/attacker.py — a deterministic, reproducible scripted attacker for the
adversarial (robustness) benchmark.

grid2op's native opponents (Geometric/RandomLine/WeightedRandom) are unreliable in
1.12 (sparse/inconsistent scheduling, crashes), so we drive the attacker from the
harness instead: a seeded schedule of "at step t, disconnect line L for d steps."
The runner fires these via the backend's /sim/attack (source="opponent"); the
defender (LLM or baseline) never sees WHO acted — only the effect in its next
observe. Fully deterministic given (seed, schedule params) -> reproducible.
"""
from __future__ import annotations
import random
from dataclasses import dataclass


# The sandbox lines the scripted attacker may hit (a spread of load-bearing lines).
ATTACK_LINES = ["0_4_1", "1_4_4", "4_5_17", "5_12_9", "8_9_10",
                "12_13_14", "3_4_6", "5_10_7"]


@dataclass
class Attack:
    start: int          # sim step the attack begins (line goes down)
    end: int            # sim step the attack ends (line restored)
    line: str           # line name to disconnect
    action_on: dict     # action dict applied at `start`
    action_off: dict    # action dict applied at `end`


def generate_attacks(seed, max_step, interval=96, duration=24,
                     lines=None, first_at=None, rng=None):
    """Deterministic attack schedule.

    Attacks at steps first_at, first_at+interval, ... up to max_step-duration.
    Each disconnects a seeded-random line from `lines` for `duration` steps.
    """
    if lines is None:
        lines = ATTACK_LINES
    if rng is None:
        rng = random.Random(seed)
    if first_at is None:
        first_at = rng.randint(interval // 2, interval)
    attacks = []
    t = first_at
    while t + duration < max_step:
        line = rng.choice(lines)
        attacks.append(Attack(
            start=t, end=t + duration, line=line,
            action_on={"set_line_status": {line: -1}},
            action_off={"set_line_status": {line: 1}},
        ))
        t += interval
    return attacks


def pending_events(attacks, new_t, prev_t):
    """Given the sim just advanced prev_t -> new_t, return the attack actions to
    fire now: any attack starting in (prev_t, new_t] fires its action_on, any
    ending in (prev_t, new_t] fires its action_off. Handles the defender
    fast-forwarding several steps between polls."""
    fires = []
    for a in attacks:
        if prev_t < a.start <= new_t:
            fires.append((a.start, "on", a.action_on, a.line))
        if prev_t < a.end <= new_t:
            fires.append((a.end, "off", a.action_off, a.line))
    fires.sort(key=lambda x: x[0])
    return fires


if __name__ == "__main__":
    atk = generate_attacks(seed=0, max_step=1200, interval=96, duration=24)
    print(f"seed=0 max_step=1200 interval=96 -> {len(atk)} attacks")
    for a in atk[:6]:
        print(f"  t={a.start:4d}..{a.end:4d}  {a.line}")
    # reproducibility
    a2 = generate_attacks(seed=0, max_step=1200, interval=96, duration=24)
    assert [ (x.start,x.line) for x in atk ] == [ (x.start,x.line) for x in a2 ], "not reproducible!"
    print("reproducible: OK")
