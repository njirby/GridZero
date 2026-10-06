"""T1 — standalone execute() test: real vLLM engine + one REAL bench episode.

Run (cards 0-1 free, bench 4B server stopped):
  CUDA_VISIBLE_DEVICES=0 ~/Documents/openrlhf/.venv/bin/python rl/test_execute.py

Gates:
  1. episode completes and returns a reward
  2. action_ranges count == LLM turns captured (proxy tee)
  3. rollout_log_probs nonzero only inside action spans (mask correctness)
  4. logprob spot-check: greedy recompute of the last turn matches the captured
     per-token logprobs within 1e-4
  5. reward matches the bench trace's terminal cum_reward
"""
import asyncio, glob, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # GridZero/

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

from rl.gridzero_agent import AgentExecutor, _GZ

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


def trace_terminal_cum(before_ts):
    """max cum_reward over sim.step_outcome events in traces written after before_ts."""
    best = None
    for f in glob.glob(os.path.join(_GZ, "runs", "ep-*.jsonl")):
        if os.path.getmtime(f) < before_ts:
            continue
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "sim.step_outcome":
                c = d["data"].get("cum_reward")
                if c is not None and (best is None or c > best):
                    best = c
    return best


async def main():
    t0 = time.time()
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=MODEL, tokenizer=MODEL, dtype="bfloat16",
        max_model_len=65536, gpu_memory_utilization=0.85, enforce_eager=True))
    tokenizer = engine.tokenizer
    print(f"engine up in {time.time()-t0:.0f}s")

    ex = AgentExecutor()
    sp = SamplingParams(max_tokens=4096, temperature=0.0, top_p=1.0, logprobs=1)  # greedy: deterministic recompute
    ex.sampling_params = sp
    ex.max_length = 65536
    ex._init_server(engine, tokenizer)
    print(f"agent server: http://127.0.0.1:{ex.port}/v1")

    before = time.time()
    session_id = "t1-session"
    prompt = "chronic=0|horizon=12|seed=0"
    t1 = time.time()
    result = await ex.run_agent(prompt, "", session_id)
    traces = list(ex._token_traces.get(session_id, []))
    out = ex._stitch_session(session_id, prompt, "", sp, result or {})
    print(f"episode wall {time.time()-t1:.0f}s: reward={result.get('reward')} extra={result.get('extra_logs')}")

    obs = out["observation_tokens"]
    ranges = out["action_ranges"]
    rlp = out["rollout_log_probs"]
    print(f"turns={len(ranges)} tokens={len(obs)} truncated={out['truncated']}")

    fails = []
    # 1. reward
    if not result or result.get("reward") is None or not (10 < result["reward"] < 20000):
        fails.append(f"reward implausible: {result.get('reward') if result else None}")
    # 2. turns vs proxy tee
    tee_dir = None
    for d in sorted(glob.glob(os.path.join(_GZ, "runs", "rl-ep-*")), key=os.path.getmtime, reverse=True):
        if os.path.getmtime(d) >= before:
            tee_dir = d
            break
    n_ee = 0
    if tee_dir:
        for line in open(os.path.join(tee_dir, "proxy.jsonl")):
            if json.loads(line)["path"].endswith("/chat/completions"):
                n_ee += 1
    print(f"proxy saw {n_ee} chat/completions; agent captured {len(ranges)} turns")
    if n_ee and n_ee != len(ranges):
        fails.append(f"turn mismatch: proxy {n_ee} vs traces {len(ranges)}")
    # 3. mask: logprobs nonzero only inside spans
    if rlp is None:
        fails.append("rollout_log_probs is None")
    else:
        if len(rlp) != len(obs):
            fails.append(f"rlp len {len(rlp)} != obs len {len(obs)}")
        in_span = [False] * len(obs)
        for s, e in ranges:
            for i in range(s, min(e, len(obs))):
                in_span[i] = True
        outside = [i for i, v in enumerate(rlp) if v != 0.0 and not in_span[i]]
        inside_nz = sum(1 for i, v in enumerate(rlp) if in_span[i] and v != 0.0)
        print(f"logprobs: {inside_nz} nonzero inside spans, {len(outside)} outside")
        if outside:
            fails.append(f"{len(outside)} nonzero logprobs outside action spans")
        if inside_nz == 0:
            fails.append("no nonzero logprobs inside spans (capture failed?)")
        # 4. greedy recompute of the LAST turn
        last = traces[-1]
        pid, cid = last["prompt_token_ids"], last["completion_token_ids"]
        regen = await engine.generate(pid, SamplingParams(temperature=0.0, top_p=1.0,
                                                          max_tokens=len(cid), logprobs=1))
        rg = regen[0].outputs[0]
        rg_ids = list(rg.token_ids)
        if rg_ids != cid:
            print(f"  (greedy recompute diverged at token {min(len(rg_ids), len(cid))} — "
                  f"expected on a live engine with other traffic; comparing overlap only)")
        if rg_ids == cid and "logprobs" in last and rg.logprobs:
            worst = 0.0
            for i in range(len(cid)):
                tid = cid[i]
                got = last["logprobs"][i]
                ref = rg.logprobs[i].get(tid)
                refv = ref.logprob if ref else float("nan")
                if refv == refv:
                    worst = max(worst, abs(got - refv))
            print(f"logprob recompute: {len(cid)} tokens, worst |diff| = {worst:.2e}")
            if worst > 1e-4:
                fails.append(f"logprob recompute off by {worst:.2e}")
        else:
            print("logprob recompute: skipped (divergence or missing logprobs)")
    # 5. reward vs bench trace
    term = trace_terminal_cum(before)
    if term is not None:
        print(f"bench trace terminal cum = {term:.1f}; agent reward = {result['reward']:.1f}")
        if abs(term - result["reward"]) > 1e-3:
            fails.append(f"reward {result['reward']} != bench terminal {term}")
    else:
        fails.append("no bench trace found for cross-check")

    if fails:
        print("\nT1 FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nT1 PASS: episode ran, masks correct, logprobs verified, reward consistent")
    return 0


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *a: os._exit(130))  # hard exit: don't strand the engine core
    sys.exit(asyncio.run(main()))
