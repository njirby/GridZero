#!/usr/bin/env python
"""Spine fixture generator.

Runs a real grid2op l2rpn_case14_sandbox, scripts a tiny episode, and emits:
  contracts/examples/grid-meta.json            (static per-env layout)
  contracts/examples/grid-state-t0.json        (C3)
  contracts/examples/grid-state-t50.json       (C3, after a line disconnect)
  contracts/examples/grid-state-t120.json      (C3, after a topology change)
  contracts/examples/render/t0000.png, t0050.png, t0120.png

The grid2op->C3 mapping below is the REFERENCE implementation. The backend
(WS A) should keep its own copy in backend/app/c3.py; keep them consistent.

Run:  ./.venv/bin/python scripts/gen_fixtures.py
"""
import json, os, warnings, datetime
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import grid2op as g2op

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "contracts", "examples")
REN = os.path.join(EX, "render")
os.makedirs(REN, exist_ok=True)

SCHEMA_VERSION = 1


def sub_types(env):
    gen_subs = set(int(s) for s in env.gen_to_subid)
    load_subs = set(int(s) for s in env.load_to_subid)
    out = []
    for i in range(env.n_sub):
        g, l = i in gen_subs, i in load_subs
        out.append("both" if (g and l) else "gen" if g else "load" if l else "other")
    return out


def build_meta(env):
    types = sub_types(env)
    layout = env.grid_layout
    tl = np.asarray(env.get_thermal_limit(), dtype=float)
    subs = []
    for i in range(env.n_sub):
        name = str(env.name_sub[i])
        x, y = layout[name]
        gp = sum(float(p) for p, s in zip(env.gen_p, env.gen_to_subid) if int(s) == i) if False else 0.0
        subs.append({"id": i, "name": name, "x": float(x), "y": float(y),
                     "type": types[i], "p": 0.0})
    lines = []
    for i in range(env.n_line):
        lines.append({
            "id": i, "name": str(env.name_line[i]),
            "or": str(env.name_sub[int(env.line_or_to_subid[i])]) if hasattr(env, "line_or_to_subid") else None,
            "ex": str(env.name_sub[int(env.line_ex_to_subid[i])]) if hasattr(env, "line_ex_to_subid") else None,
            "thermal_limit": float(tl[i]),
        })
    gens = []
    for i in range(env.n_gen):
        gens.append({"id": i, "name": str(env.name_gen[i]),
                     "sub": str(env.name_sub[int(env.gen_to_subid[i])])})
    return {"schema_version": SCHEMA_VERSION, "env": "l2rpn_case14_sandbox",
            "n_line": int(env.n_line), "n_sub": int(env.n_sub),
            "n_gen": int(env.n_gen), "n_load": int(env.n_load),
            "delta_min": 5.0, "subs": subs, "lines": lines, "gens": gens}


def to_c3(env, obs, reward, cum_reward, last_action, png):
    tl = np.asarray(env.get_thermal_limit(), dtype=float)
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
            "rho": round(float(rho[i]), 4),
            "p_or": round(float(p_or[i]), 3), "p_ex": round(float(p_ex[i]), 3),
            "status": st, "overflow": bool(rho[i] > 1.0),
            "cooldown": int(cooldown[i]), "maint": int(maint[i]),
        })

    subs = []
    gen_p = np.asarray(obs.gen_p, dtype=float)
    load_p = np.asarray(obs.load_p, dtype=float)
    for i in range(env.n_sub):
        name = str(env.name_sub[i])
        gp = float(sum(gen_p[j] for j, s in enumerate(env.gen_to_subid) if int(s) == i))
        lp = float(sum(load_p[j] for j, s in enumerate(env.load_to_subid) if int(s) == i))
        x, y = layout[name]
        subs.append({"id": i, "name": name, "x": float(x), "y": float(y),
                     "type": types[i], "p": round(gp - lp, 3)})

    gens = []
    for i in range(env.n_gen):
        gens.append({"id": i, "name": str(env.name_gen[i]),
                     "sub": str(env.name_sub[int(env.gen_to_subid[i])]),
                     "p": round(float(gen_p[i]), 3),
                     "renewable": bool(obs.gen_renewable[i]) if hasattr(obs, "gen_renewable") else False,
                     "redispatchable": bool(env.gen_redispatchable[i]) if hasattr(env, "gen_redispatchable") else True})

    n_down = int(sum(1 for l in lines if l["status"] == "down"))
    n_overflow = int(sum(1 for l in lines if l["overflow"]))
    return {
        "schema_version": SCHEMA_VERSION, "env": "l2rpn_case14_sandbox",
        "t": int(obs.current_step), "max_t": int(obs.max_step),
        "reward": round(float(reward), 4), "cum_reward": round(float(cum_reward), 4),
        "done": False, "cause": None, "delta_min": float(obs.delta_time),
        "sim_clock": str(obs.get_time_stamp()),
        "n_line": int(env.n_line), "n_sub": int(env.n_sub), "n_gen": int(env.n_gen),
        "max_rho": round(float(np.max(rho)), 4), "n_down": n_down, "n_overflow": n_overflow,
        "lines": lines, "subs": subs, "gens": gens, "alarms": [],
        "last_action": last_action, "png": png,
    }


def main():
    env = g2op.make("l2rpn_case14_sandbox")
    env.attach_renderer()
    meta = build_meta(env)
    json.dump(meta, open(os.path.join(EX, "grid-meta.json"), "w"), indent=2)

    A = env.action_space
    obs = env.reset()
    # Harness reward semantics (see C3): `reward` = most recent step's reward
    # (at t=0 the reset/initial reward); `cum_reward` = sum of COMPLETED step
    # rewards only, so it starts at 0 at t=0 and does NOT include the reset value.
    reward = env.current_reward
    cum = 0.0

    def render(t):
        env.render()
        out = os.path.join(REN, f"t{int(obs.current_step):04d}.png")
        env.viewer_fig.savefig(out, format="png", dpi=90, bbox_inches="tight")
        return os.path.relpath(out, EX)

    # t=0
    c3 = to_c3(env, obs, reward, cum, None, render(0))
    json.dump(c3, open(os.path.join(EX, "grid-state-t0.json"), "w"), indent=2)

    # advance to t=49 with no-ops
    for _ in range(49):
        obs, r, d, info = env.step(A())
        reward, cum = r, cum + r
    # at t=49 apply a disconnect: line index of "0_4_1"
    li = int(list(env.name_line).index("0_4_1"))
    act = A()
    act = A.disconnect_powerline(line_id=li, previous_action=act)
    obs, r, d, info = env.step(act)
    reward, cum = r, cum + r   # now t=50
    la = {"source": "auto", "summary": f"set_line_status {env.name_line[li]}=down",
          "args": {"set_line_status": {str(env.name_line[li]): -1}}, "t": int(obs.current_step) - 1}
    c3 = to_c3(env, obs, reward, cum, la, render(50))
    json.dump(c3, open(os.path.join(EX, "grid-state-t50.json"), "w"), indent=2)

    # advance to t=119
    for _ in range(69):
        obs, r, d, info = env.step(A())
        reward, cum = r, cum + r
    # apply a topology change: change_bus on a line extremity
    ln2 = str(env.name_line[4])
    act = A()
    act = A.change_bus(ln2, extremity="or", previous_action=act)
    obs, r, d, info = env.step(act)
    reward, cum = r, cum + r   # t=120
    la = {"source": "auto", "summary": f"change_bus {ln2} or",
          "args": {"change_bus": {"lines_or_id": {ln2: True}}}, "t": int(obs.current_step) - 1}
    c3 = to_c3(env, obs, reward, cum, la, render(120))
    json.dump(c3, open(os.path.join(EX, "grid-state-t120.json"), "w"), indent=2)

    print("wrote:")
    for f in ["grid-meta.json", "grid-state-t0.json", "grid-state-t50.json", "grid-state-t120.json"]:
        p = os.path.join(EX, f)
        print(" ", f, os.path.getsize(p), "bytes")
    print("  pngs:", os.listdir(REN))
    print("t50 max_rho", json.load(open(os.path.join(EX, 'grid-state-t50.json')))["max_rho"],
          "n_down", json.load(open(os.path.join(EX, 'grid-state-t50.json')))["n_down"])


if __name__ == "__main__":
    main()
