"""Read gridtrace sidecar JSONL and stitch it into one OpenRLHF rollout row.

The plugin (rl/gridtrace_plugin.js) writes one JSONL line per LLM request:
  {ts, session, model, mode: "full"|"delta", prompt_token_ids,
   completion_token_ids, logprobs, finish_reason, usage, n_tools}

Prompt ids are delta-encoded: after the first turn each record stores only the
NEW tokens (previous completion + tool-result rendering). A "full" record means
the prefix chain broke (opencode compaction / session restart) and the chain
restarts there.

OpenRLHF's agent contract (openrlhf/trainer/ppo_utils/samples_generator.py)
consumes ONE row per execute() call: a flat `observation_tokens` sequence plus
`action_ranges` (spans of model-generated tokens) plus `rollout_log_probs`
(aligned to observation_tokens, 0.0 off-span). We emit the LAST chain
(post the most recent compaction) as that single row.
"""
from __future__ import annotations
import json, os, glob


def read_sidecar(trace_dir: str) -> list[dict]:
    """Agent turns for the session(s) in `trace_dir`, in request order.

    Aux calls (opencode title generation etc.) carry no tools and are dropped:
    they are separate conversations, not part of the episode chain."""
    recs = []
    for f in sorted(glob.glob(os.path.join(trace_dir, "ses_*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("n_tools", 0) > 0:
                    recs.append(r)
    recs.sort(key=lambda r: r.get("ts", 0))
    return recs


def _chain_start(turns: list[dict]) -> int:
    """Index of the last 'full' record (start of the final chain)."""
    start = 0
    for i, t in enumerate(turns):
        if t.get("mode") == "full":
            start = i
    return start


def stitch(turns: list[dict]) -> dict:
    """Stitch the final chain into one OpenRLHF rollout row."""
    if not turns:
        raise ValueError("no sidecar records to stitch")
    start = _chain_start(turns)
    seg = turns[start:]

    prefix: list[int] = []
    action_ranges: list[tuple[int, int]] = []
    for i, t in enumerate(seg):
        if i == 0:
            prefix = list(t["prompt_token_ids"])          # full P_i
        else:
            prefix = prefix + t["prompt_token_ids"]        # P_i = P_{i-1} + D_i
        s = len(prefix)                                     # offset of C_i in the final seq
        c = t["completion_token_ids"]
        action_ranges.append((s, s + len(c)))

    obs = prefix + list(seg[-1]["completion_token_ids"])    # P_last + C_last
    lp = [0.0] * len(obs)
    for (s, e), t in zip(action_ranges, seg):
        for j, v in enumerate(t.get("logprobs", [])):
            if s + j < len(lp):
                lp[s + j] = v

    return {
        "observation_tokens": obs,
        "action_ranges": action_ranges,
        "rollout_log_probs": lp,
        "truncated": seg[-1].get("finish_reason") == "length",
    }


def rollout_from_sidecar(trace_dir: str) -> dict | None:
    turns = read_sidecar(trace_dir)
    if not turns:
        return None
    row = stitch(turns)
    row["n_turns"] = len(turns)
    row["n_turns_used"] = len(turns) - _chain_start(turns)
    row["n_action_tokens"] = sum(e - s for s, e in row["action_ranges"])
    return row
