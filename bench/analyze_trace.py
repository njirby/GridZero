#!/usr/bin/env python
"""bench/analyze_trace.py — extract the model's behavior from an episode trace.

The backend bus persists every event to runs/<ep>.jsonl. This parses it into a
behavior breakdown: how many sim steps the agent took, the split of its `simctl`
actions by type (redispatch / set_line_status / change_bus / no-op / other), how
many times it observed/rendered/did other tool work, and a per-step action list.
Enriches an EpisodeResult's simctl_acts/observes/renders + a behavior field.

Usage:
  ./.venv/bin/python bench/analyze_trace.py --results runs/llm-0-.../results.json
  ./.venv/bin/python bench/analyze_trace.py --ep runs/ep-....jsonl
"""
import argparse, json, os, re, sys
from collections import Counter


def _cmd_of(input):
    if isinstance(input, dict):
        c = input.get("command", "")
        if isinstance(c, list):
            c = " ".join(c)
        return str(c)
    return str(input)


def analyze_trace(ep_path):
    """Return a behavior dict from a runs/<ep>.jsonl trace."""
    steps = []            # sim.step_outcome where source==agent
    tool_cmds = []        # (tool, command) for agent tool_results
    n_delta_text = n_delta_reason = 0
    n_turns = 0
    for line in open(ep_path):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t, d = ev.get("type"), ev.get("data", {})
        if t == "sim.step_outcome" and d.get("source") == "agent":
            a = d.get("action") or {}
            steps.append({"t": d.get("t"), "args": a.get("args"), "summary": a.get("summary"),
                          "illegal": d.get("illegal"), "ambiguous": d.get("ambiguous"),
                          "new_overloads": d.get("new_overloads")})
        elif t == "agent.tool_result":
            # recover the command from the input (carried by our driver) or output
            inp = d.get("input") or {}
            cmd = _cmd_of(inp)
            tool_cmds.append((d.get("tool", "?"), cmd))
        elif t == "agent.delta":
            if d.get("field") == "reasoning":
                n_delta_reason += 1
            else:
                n_delta_text += 1
        elif t == "agent.turn_end":
            n_turns += 1

    # classify the agent's simctl actions
    action_types = Counter()
    for s in steps:
        args = s.get("args") or {}
        if not args:
            action_types["no-op (empty act / step)"] += 1
        else:
            for k in args:
                action_types[k] += 1
    # classify all bash commands the agent ran (by first token)
    bash_kinds = Counter()
    for tool, cmd in tool_cmds:
        if tool == "bash":
            m = re.match(r"\s*(\S+)", cmd)
            first = m.group(1) if m else "?"
            if first == "simctl":
                m2 = re.search(r"\bsimctl\s+(\w+)", cmd)
                bash_kinds["simctl:" + (m2.group(1) if m2 else "?")] += 1
            else:
                bash_kinds[first.split("/")[-1]] += 1
        else:
            bash_kinds[tool] += 1

    n_obs = bash_kinds.get("simctl:observe", 0)
    n_act = bash_kinds.get("simctl:act", 0)
    n_step = bash_kinds.get("simctl:step", 0)
    n_render = bash_kinds.get("simctl:render", 0)

    return {
        "ep": os.path.basename(ep_path),
        "agent_sim_steps": len(steps),
        "n_llm_turns": n_turns,
        "simctl_act_calls": n_act,
        "simctl_observe_calls": n_obs,
        "simctl_step_calls": n_step,
        "simctl_render_calls": n_render,
        "other_tool_calls": sum(v for k, v in bash_kinds.items() if not k.startswith("simctl:")),
        "tool_command_breakdown": dict(bash_kinds),
        "agent_action_key_breakdown": dict(action_types),
        "illegal_acts": sum(1 for s in steps if s.get("illegal")),
        "steps_with_new_overload": sum(1 for s in steps if s.get("new_overloads")),
        "n_delta_text": n_delta_text, "n_delta_reason": n_delta_reason,
        "action_sequence": steps,
    }


def backfill_results(results_path):
    """Read a results.json, analyze each episode's trace, write behavior + counts back."""
    res = json.load(open(results_path))
    changed = False
    for r in res:
        ep = r.get("ep", "")
        ep_path = os.path.join(os.path.dirname(results_path), "..", ep + ".jsonl") \
            if not ep.startswith("runs/") else os.path.join(os.path.dirname(results_path), "..", ep + ".jsonl")
        # ep is like 'ep-2026...'; trace lives in runs/<ep>.jsonl
        cand = os.path.join(os.path.dirname(os.path.dirname(results_path)), "runs", ep + ".jsonl")
        if not os.path.exists(cand):
            cand = os.path.join(os.path.dirname(os.path.dirname(results_path)), ep + ".jsonl")
        if os.path.exists(cand):
            b = analyze_trace(cand)
            r["simctl_acts"] = b["simctl_act_calls"]
            r["simctl_observes"] = b["simctl_observe_calls"]
            r["simctl_renders"] = b["simctl_render_calls"]
            r["llm_turns"] = b["n_llm_turns"]
            r["behavior"] = {k: v for k, v in b.items() if k not in ("action_sequence",)}
            r["action_sequence"] = b["action_sequence"]
            changed = True
        else:
            print(f"  (no trace found for ep {ep}; skipped)")
    if changed:
        json.dump(res, open(results_path, "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="a run_llm results.json to backfill")
    ap.add_argument("--ep", default=None, help="a runs/<ep>.jsonl to analyze directly")
    a = ap.parse_args()
    if a.results:
        res = backfill_results(a.results)
        for r in res:
            b = r.get("behavior", {})
            print(f"chronic {r['chronic']}: survived {r['survived']}/{r['horizon']} "
                  f"cum {r['cum_reward']:.0f} | acts={r.get('simctl_acts')} "
                  f"observes={r.get('simctl_observes')} renders={r.get('simctl_renders')} "
                  f"turns={r.get('llm_turns')} | action_keys={b.get('agent_action_key_breakdown')}")
        print(f"backfilled {a.results}")
    elif a.ep:
        b = analyze_trace(a.ep)
        json.dump(b, sys.stdout, indent=2)
    else:
        print("give --results <results.json> or --ep <ep.jsonl>")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
