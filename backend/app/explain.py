"""explain.py — turn grid2op's action failures into precise, factual messages.

Two failure points, with different consequences (stated in every message):
- prevalidate(): the action dict cannot be built (bad shape, unknown element,
  out-of-range value). Nothing is applied and time does NOT advance.
- explain_step(): grid2op stepped but rejected the action (illegal / ambiguous /
  infeasible redispatch). The whole action was replaced by do-nothing and the
  step DID advance (reward 0).
Messages name the element, what was requested and the limit that was violated,
using the pre-step observation for current values. No advice.
"""
from __future__ import annotations
import difflib, re

import numpy as np

# action key -> what its names refer to ("line" / "gen" / "storage"), and the
# value shape: "dict" (name -> number) or "list" (names)
_FLAT = {"set_line_status": ("line", "dict"), "change_line_status": ("line", "list"),
         "redispatch": ("gen", "dict"), "curtail": ("gen", "dict"),
         "set_storage": ("storage", "dict")}
_BUS_SUBKEYS = {"lines_or_id": "line", "lines_ex_id": "line", "loads_id": "load",
                "generators_id": "gen", "storages_id": "storage"}


def _names(env, kind):
    return [str(n) for n in {"line": env.name_line, "gen": env.name_gen, "load": env.name_load,
                             "storage": getattr(env, "name_storage", [])}[kind]]


_KIND_WORD = {"line": "line", "gen": "generator", "load": "load", "storage": "storage unit"}


def _unknown(env, kind, name):
    pool = _names(env, kind)
    if not pool:
        return f"this grid has no {_KIND_WORD[kind]}s (got {name!r})"
    close = difflib.get_close_matches(str(name), pool, n=3, cutoff=0.5)
    hint = f"; closest names: {', '.join(close)}" if close else ""
    return f"unknown {_KIND_WORD[kind]} {name!r}{hint} (`simctl list {kind}s` lists them)" \
        if kind in ("line", "gen") else f"unknown {_KIND_WORD[kind]} {name!r}{hint}"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def prevalidate(env, action: dict):
    """Return a message if the action dict is malformed, else None."""
    for key, val in action.items():
        if key in _FLAT:
            kind, shape = _FLAT[key]
            if shape == "dict" and not isinstance(val, dict):
                return f"`{key}` must be an object {{\"<{kind} name>\": value}}, got {type(val).__name__}"
            if shape == "list" and not isinstance(val, list):
                return f"`{key}` must be a list of {kind} names, got {type(val).__name__}"
            known = set(_names(env, kind))
            for name in val:
                if str(name) not in known:
                    return _unknown(env, kind, name)
            if shape == "dict":
                for name, v in val.items():
                    x = _num(v)
                    if x is None:
                        return f"`{key}` value for {name} must be a number, got {v!r}"
                    if key == "set_line_status" and x not in (-1, 0, 1):
                        return f"set_line_status value for {name} must be 1 (connect), -1 (disconnect) or 0, got {v!r}"
                    if key == "curtail" and not 0.0 <= x <= 1.0:
                        return f"curtail value for {name} must be between 0.0 and 1.0 (fraction of pmax), got {v!r}"
        elif key in ("set_bus", "change_bus"):
            if not isinstance(val, dict):
                return f"`{key}` must be an object keyed by {', '.join(_BUS_SUBKEYS)}"
            for sub, items in val.items():
                if sub not in _BUS_SUBKEYS:
                    return f"`{key}` has unknown sub-key {sub!r} (valid: {', '.join(_BUS_SUBKEYS)})"
                kind = _BUS_SUBKEYS[sub]
                if key == "change_bus" and not isinstance(items, list):
                    return f"`change_bus.{sub}` must be a list of {kind} names"
                if key == "set_bus" and not isinstance(items, dict):
                    return f"`set_bus.{sub}` must be an object {{\"<{kind} name>\": bus}}"
                known = set(_names(env, kind))
                for name in items:
                    if str(name) not in known:
                        return _unknown(env, kind, name)
                if key == "set_bus":
                    for name, b in items.items():
                        if _num(b) not in (-1, 0, 1, 2):
                            return f"set_bus.{sub} value for {name} must be 1 or 2 (busbar), -1 (disconnect) or 0, got {b!r}"
    return None


def not_built_suffix(t):
    return f"Not applied; time did NOT advance (t={t})."


def stepped_suffix(t, reward):
    return f"Not applied: the step advanced to t={t} with do-nothing instead (reward {reward:g})."


def _ids_after(msg, word):
    """Integer ids from the first [...] after `word` in a grid2op message."""
    m = re.search(word + r"[^\[]*\[([^\]]*)\]", msg)
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))] if m else []


def _redisp_explain(env, prev_obs, action, msg):
    out = []
    redisp = {str(k): float(v) for k, v in (action.get("redispatch") or {}).items()}
    names = list(_names(env, "gen"))
    target = np.asarray(prev_obs.target_dispatch, dtype=float)
    gen_p = np.asarray(prev_obs.gen_p, dtype=float)
    for name, d in redisp.items():
        i = names.index(name)
        up, down = float(env.gen_max_ramp_up[i]), float(env.gen_max_ramp_down[i])
        span = float(env.gen_pmax[i] - env.gen_pmin[i])
        if abs(d) < 1e-7:
            continue
        if not env.gen_redispatchable[i]:
            out.append(f"{name} is not redispatchable (redispatchable gens: "
                       f"`simctl list gens --where redispatchable=true`)")
        elif "ramp up" in msg and d > up:
            out.append(f"redispatch {name} {d:+g} MW exceeds its ramp-up limit {up:g} MW/step "
                       f"(current target_dispatch {target[i]:g} MW)")
        elif "ramp down" in msg and -d > down:
            out.append(f"redispatch {name} {d:+g} MW exceeds its ramp-down limit {down:g} MW/step "
                       f"(current target_dispatch {target[i]:g} MW)")
        elif "pmax - pmin" in msg and target[i] + d > span:
            out.append(f"redispatch {name} {d:+g} MW would make its target_dispatch {target[i] + d:g} MW, "
                       f"above pmax - pmin = {span:g} MW (pmax {env.gen_pmax[i]:g}, pmin {env.gen_pmin[i]:g})")
        elif "pmin - pmax" in msg and target[i] + d < -span:
            out.append(f"redispatch {name} {d:+g} MW would make its target_dispatch {target[i] + d:g} MW, "
                       f"below pmin - pmax = {-span:g} MW (pmax {env.gen_pmax[i]:g}, pmin {env.gen_pmin[i]:g})")
        elif "turned off" in msg and abs(gen_p[i]) <= 1e-7:
            out.append(f"{name} is producing 0 MW (turned off); a turned-off generator cannot be redispatched")
        elif "above pmax" in msg and gen_p[i] + d > env.gen_pmax[i]:
            out.append(f"redispatch {name} {d:+g} MW on top of its {gen_p[i]:g} MW output exceeds pmax "
                       f"{env.gen_pmax[i]:g} MW")
        elif "below pmin" in msg and gen_p[i] + d < env.gen_pmin[i]:
            out.append(f"redispatch {name} {d:+g} MW on top of its {gen_p[i]:g} MW output is below pmin "
                       f"{env.gen_pmin[i]:g} MW")
    return out


def _one(env, prev_obs, action, exc):
    cls = type(exc).__name__
    msg = re.sub(r"np\.\w+\((-?[\d.]+)\)", r"\1", str(exc))  # "np.int64(93)" -> "93"
    p = env.parameters
    if cls in ("InvalidRedispatching", "IllegalRedispatching"):
        got = _redisp_explain(env, prev_obs, action, msg)
        if got:
            return "; ".join(got)
    if cls == "InvalidCurtailment":
        names = _names(env, "gen")
        bad = [str(n) for n in (action.get("curtail") or {}) if not env.gen_renewable[names.index(str(n))]]
        if "non renewable" in msg and bad:
            return (f"{', '.join(bad)} {'is' if len(bad) == 1 else 'are'} not renewable; only renewable gens "
                    f"can be curtailed (`simctl list gens --where renewable=true`)")
    if cls == "InvalidStorage":
        names = _names(env, "storage")
        out = []
        for n, v in (action.get("set_storage") or {}).items():
            i, v = names.index(str(n)), float(v)
            if v > env.storage_max_p_absorb[i]:
                out.append(f"set_storage {n} {v:g} MW exceeds its max charging power "
                           f"{env.storage_max_p_absorb[i]:g} MW (positive = charge)")
            elif -v > env.storage_max_p_prod[i]:
                out.append(f"set_storage {n} {v:g} MW exceeds its max discharging power "
                           f"{env.storage_max_p_prod[i]:g} MW (negative = discharge)")
        if out:
            return "; ".join(out)
    if "Powerline with ids" in msg and "cooldown" in msg:
        ids, cds = _ids_after(msg, "ids"), _ids_after(msg, "cooldown of")
        parts = [f"line {env.name_line[i]} cannot be changed for {c} more step{'s' * (c != 1)}"
                 for i, c in zip(ids, map(int, cds))]
        return ("; ".join(parts) + f" (cooldown: {p.NB_TIMESTEP_COOLDOWN_LINE} steps after an action on it, "
                f"{p.NB_TIMESTEP_RECONNECTION} after a trip)")
    if "Substation with ids" in msg and "cooldown" in msg:
        ids, cds = _ids_after(msg, "ids"), _ids_after(msg, "cooldown of")
        parts = [f"substation {env.name_sub[i]} cannot be changed for {c} more step{'s' * (c != 1)}"
                 for i, c in zip(ids, map(int, cds))]
        return "; ".join(parts) + f" (cooldown: {p.NB_TIMESTEP_COOLDOWN_SUB} steps after a topology change)"
    if msg.find("More than") >= 0 and ("line status" in msg or "substation" in msg):
        is_line = "line status" in msg
        groups = [[int(x) for x in re.findall(r"\d+", g)]
                  for g in re.findall(r"\[([^\[\]]*)\]", msg.split(":", 1)[-1])]
        limit = int(p.MAX_LINE_STATUS_CHANGED if is_line else p.MAX_SUB_CHANGED)
        groups = [g for g in groups if len(g) > limit] or groups
        names = env.name_line if is_line else env.name_sub
        what = "line status change" if is_line else "substation change"
        scope = " in each area" if "area" in msg else ""
        got = "; ".join(f"got {len(g)}: {', '.join(str(names[i]) for i in g)}" for g in groups)
        return f"at most {limit} {what}{'s' * (limit != 1)} per action{scope} ({got})"
    if "reconnect" in msg.lower():
        return re.sub(r'^.*?"', "", msg).rstrip('"') or msg
    # fallback: grid2op's own text without the exception-class prefix
    text = re.sub(r"^(Grid2OpException\s+)?(\w+\s+)*?\"", "", msg).rstrip('"')
    return f"{cls}: {text or msg}"


def explain_step(env, prev_obs, action, info):
    """If grid2op rejected the action on this step, return (kind, explanation),
    kind in {"Illegal action", "Ambiguous action"}; else None."""
    from grid2op.Exceptions import AmbiguousAction, IllegalAction
    excs = [e for e in (info.get("exception") or []) if isinstance(e, (AmbiguousAction, IllegalAction))]
    if not excs and not (info.get("is_illegal") or info.get("is_ambiguous")):
        return None
    kind = "Ambiguous action" if (info.get("is_ambiguous") or
                                  any(isinstance(e, AmbiguousAction) for e in excs)) else "Illegal action"
    if not excs:
        return kind, "rejected by grid2op (no reason given)"
    whys = []
    for e in excs:
        try:
            whys.append(_one(env, prev_obs, action, e))
        except Exception:  # never let the explainer break a step: fall back to grid2op's text
            whys.append(f"{type(e).__name__}: {e}")
    return kind, "; ".join(dict.fromkeys(whys))
