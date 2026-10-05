"""c3.py — canonical grid2op -> GRID-STATE (C3) mapping.

Mirrors the validated logic in scripts/gen_fixtures.py. The backend keeps its
own copy here; the two must stay consistent.
"""
from __future__ import annotations
import numpy as np

SCHEMA_VERSION = 1


def _f(x, d=0.0):
    """float() with NaN/None -> default (down lines can report NaN loading)."""
    try:
        x = float(x)
        return d if x != x else x
    except (TypeError, ValueError):
        return d


def _env_name(env) -> str:
    # grid2op's env.name carries the backend suffix (e.g. "...PandaPowerBackend")
    return str(env.name).split("PandaPowerBackend")[0]


def sub_types(env):
    gen_subs = set(int(s) for s in env.gen_to_subid)
    load_subs = set(int(s) for s in env.load_to_subid)
    out = []
    for i in range(env.n_sub):
        g, l = i in gen_subs, i in load_subs
        out.append("both" if (g and l) else "gen" if g else "load" if l else "other")
    return out


def build_meta(env) -> dict:
    """Static per-env layout (coords, types, thermal limits). Sent once to the UI."""
    types = sub_types(env)
    layout = env.grid_layout
    tl = np.asarray(env.get_thermal_limit(), dtype=float)
    subs = []
    for i in range(env.n_sub):
        name = str(env.name_sub[i])
        x, y = layout[name]
        subs.append({"id": i, "name": name, "x": float(x), "y": float(y),
                     "type": types[i], "p": 0.0})
    lines = []
    for i in range(env.n_line):
        or_sub = str(env.name_sub[int(env.line_or_to_subid[i])])
        ex_sub = str(env.name_sub[int(env.line_ex_to_subid[i])])
        lines.append({"id": i, "name": str(env.name_line[i]), "or": or_sub, "ex": ex_sub,
                      "thermal_limit": float(tl[i])})
    gens = []
    for i in range(env.n_gen):
        gens.append({"id": i, "name": str(env.name_gen[i]),
                     "sub": str(env.name_sub[int(env.gen_to_subid[i])])})
    return {"schema_version": SCHEMA_VERSION, "env": _env_name(env),
            "n_line": int(env.n_line), "n_sub": int(env.n_sub),
            "n_gen": int(env.n_gen), "n_load": int(env.n_load),
            "delta_min": 5.0, "subs": subs, "lines": lines, "gens": gens}


def to_c3(env, obs, reward, cum_reward, last_action=None, png="") -> dict:
    """Build a full GRID-STATE (C3) from a live obs.

    NOTE: only reads static env attrs (name_*, grid_layout, *_to_subid) + obs
    arrays — never env.get_thermal_limit() (that raises after the episode is
    done and the C3 state doesn't need it; thermal limits live in build_meta).
    """
    layout = env.grid_layout
    types = sub_types(env)
    line_status = np.asarray(obs.line_status, dtype=bool)
    rho = np.asarray(obs.rho, dtype=float)
    cooldown = np.asarray(obs.time_before_cooldown_line, dtype=int)
    maint = np.asarray(obs.time_next_maintenance, dtype=int)
    p_or = np.asarray(obs.p_or, dtype=float)
    p_ex = np.asarray(obs.p_ex, dtype=float)

    lines = []
    for i in range(env.n_line):
        if not line_status[i]:
            st = "down"
        elif cooldown[i] > 0:
            st = "cooldown"
        elif maint[i] == 0:
            st = "maintenance"
        else:
            st = "up"
        lines.append({
            "id": i, "name": str(env.name_line[i]),
            "or": str(env.name_sub[int(obs.line_or_to_subid[i])]),
            "ex": str(env.name_sub[int(obs.line_ex_to_subid[i])]),
            "rho": round(_f(rho[i]), 4),
            "p_or": round(_f(p_or[i]), 3), "p_ex": round(_f(p_ex[i]), 3),
            "status": st, "overflow": bool(_f(rho[i]) > 1.0),
            "cooldown": int(cooldown[i]), "maint": int(maint[i]),
        })

    gen_p = np.asarray(obs.gen_p, dtype=float)
    load_p = np.asarray(obs.load_p, dtype=float)
    subs = []
    for i in range(env.n_sub):
        name = str(env.name_sub[i])
        gp = float(sum(gen_p[j] for j, s in enumerate(env.gen_to_subid) if int(s) == i))
        lp = float(sum(load_p[j] for j, s in enumerate(env.load_to_subid) if int(s) == i))
        x, y = layout[name]
        subs.append({"id": i, "name": name, "x": _f(x), "y": _f(y),
                     "type": types[i], "p": round(_f(gp - lp), 3)})

    gens = []
    for i in range(env.n_gen):
        gens.append({"id": i, "name": str(env.name_gen[i]),
                     "sub": str(env.name_sub[int(env.gen_to_subid[i])]),
                     "p": round(_f(gen_p[i]), 3),
                     "renewable": bool(obs.gen_renewable[i]) if hasattr(obs, "gen_renewable") else False,
                     "redispatchable": bool(env.gen_redispatchable[i]) if hasattr(env, "gen_redispatchable") else True})

    n_down = int(sum(1 for l in lines if l["status"] == "down"))
    n_overflow = int(sum(1 for l in lines if l["overflow"]))
    max_rho = max((_f(rho[i]) for i in range(env.n_line)), default=0.0)
    return {
        "schema_version": SCHEMA_VERSION, "env": _env_name(env),
        "t": int(obs.current_step), "max_t": int(obs.max_step),
        "reward": round(_f(reward), 4), "cum_reward": round(_f(cum_reward), 4),
        "done": bool(obs.current_step >= obs.max_step), "cause": None,
        "delta_min": _f(obs.delta_time, 5.0), "sim_clock": str(obs.get_time_stamp()),
        "n_line": int(env.n_line), "n_sub": int(env.n_sub), "n_gen": int(env.n_gen),
        "max_rho": round(max_rho, 4), "n_down": n_down, "n_overflow": n_overflow,
        "lines": lines, "subs": subs, "gens": gens, "alarms": [],
        "last_action": last_action, "png": png,
    }
