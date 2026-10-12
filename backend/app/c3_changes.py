"""c3_changes.py — the `observe --changes` view: a C3 filtered to the entities
that changed since the last view the backend served to this viewer.

Change rules (mechanical; documented in contracts/C2-sim-api.md):
- An entity (line / sub / gen) is reported IN FULL (all its C3 fields) when ANY
  of its fields changed vs the version of that entity the viewer was last shown.
- Discrete fields are compared exactly: line status/or/ex/overflow/or_bus/
  ex_bus/timestep_overflow, gen connected, gen and sub static fields. Two countdown fields only count at state transitions, not on
  every tick: line `cooldown` (0 <-> >0, or a restart = value went up) and line
  `maint` (-1 none / 0 now / >0 scheduled).
- Continuous fields count when |delta| >= a threshold: line rho >= rho_eps;
  gen p / target_dispatch / actual_dispatch and sub p >= mw_eps. Line p_or/p_ex are carried along but do not
  trigger (rho is the normalized measure of the same flow).

The per-viewer baseline is updated only for the entities it was shown, so slow
drift accumulates against what the viewer last saw and is reported once it
crosses the threshold — the viewer's picture is never off by more than eps.
"""
from __future__ import annotations

import os

# Defaults are per deployment (SIM_CHANGES_RHO_EPS / SIM_CHANGES_MW_EPS); callers may override per call.
DEFAULT_RHO_EPS = float(os.environ.get("SIM_CHANGES_RHO_EPS") or 0.01)
DEFAULT_MW_EPS = float(os.environ.get("SIM_CHANGES_MW_EPS") or 1.0)

ENTITY_KEYS = ("lines", "subs", "gens")
# fields compared with a threshold: {kind: {field: which eps}}
_CONT = {"lines": {"rho": "rho_eps"}, "subs": {"p": "mw_eps"},
         "gens": {"p": "mw_eps", "target_dispatch": "mw_eps", "actual_dispatch": "mw_eps"}}
# carried along, never a trigger on their own
_IGNORED = {"lines": {"p_or", "p_ex"}}


def _maint_state(m):
    return "none" if m < 0 else "now" if m == 0 else "scheduled"


def _changed(kind, old, new, eps) -> bool:
    for k, v in new.items():
        ov = old.get(k)
        if k in _IGNORED.get(kind, ()):
            continue
        if k in _CONT.get(kind, {}):
            if ov is None or abs(float(v) - float(ov)) >= eps[_CONT[kind][k]]:
                return True
        elif kind == "lines" and k == "cooldown":
            ov = ov or 0
            if (ov == 0) != (v == 0) or v > ov:
                return True
        elif kind == "lines" and k == "maint":
            if ov is None or _maint_state(ov) != _maint_state(v):
                return True
        elif v != ov:
            return True
    return False


def new_view(c3: dict) -> dict:
    """Baseline after serving the FULL c3 to a viewer."""
    return {"t": c3.get("t", 0),
            **{k: {e["id"]: e for e in c3.get(k, [])} for k in ENTITY_KEYS}}


def changes(c3: dict, view: dict | None, rho_eps=DEFAULT_RHO_EPS, mw_eps=DEFAULT_MW_EPS):
    """Return (changes_dict, updated_view). view=None -> full dump."""
    eps = {"rho_eps": float(rho_eps), "mw_eps": float(mw_eps)}
    out = {k: v for k, v in c3.items() if k not in ENTITY_KEYS}
    full = view is None
    if full:
        view = {"t": None, **{k: {} for k in ENTITY_KEYS}}
    out["full"] = full
    out["changes_since_t"] = view["t"]
    out["thresholds"] = eps
    new_view_ = {"t": c3.get("t", 0)}
    for kind in ENTITY_KEYS:
        seen = dict(view[kind])
        picked = []
        for e in c3.get(kind, []):
            old = seen.get(e["id"])
            if full or old is None or _changed(kind, old, e, eps):
                picked.append(e)
                seen[e["id"]] = e
        out[kind] = picked
        out[f"{kind}_changed"] = len(picked)
        new_view_[kind] = seen
    return out, new_view_
