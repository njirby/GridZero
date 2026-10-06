"""RL agent executor for GridZero bench episodes (OpenRLHF `--train.agent_func_path`).

Fork of OpenRLHF's examples/python/agent_func_openai_server_executor.py with:
  - the request's `tools` passed into the chat-template render (opencode sends
    tool definitions per request; without this the model never sees them),
  - the server bound to 0.0.0.0 (agents run in per-episode netns and reach it
    via the gateway IP),
  - run_agent() driving a REAL bench episode (bench.run_llm.run_episode):
    backend + netns + bwrap-sandboxed opencode, pointed at this server (the
    policy model). Token capture is on the opencode side: the gridtrace plugin
    (rl/gridtrace_plugin.js, enabled via GRZ_RL_TRACE) tees every request into
    a per-port sidecar with authoritative token ids + logprobs; execute()
    stitches that sidecar (rl/sidecar.py) into the OpenRLHF rollout row.
    opencode itself is unmodified stock.

Dataset prompt format: "chronic=<int>|horizon=<int>|seed=<int>"
Env: GRZ_RL_MODEL_NAME (opencode model id, default qwen3.5-4b),
     OPENCODE_PROVIDER (default vllm4b), OPENCODE_NETNS (default 1),
     GRZ_RL_TRACE (set by run_agent; enables the plugin capture).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
from urllib.request import urlopen
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from openai import AsyncOpenAI
from vllm import SamplingParams

from openrlhf.utils.agent import AgentExecutorBase

logging.basicConfig()
logger = logging.getLogger(__name__)

_GZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # GridZero/
if _GZ not in sys.path:
    sys.path.insert(0, _GZ)
from bench import run_llm  # noqa: E402
from rl.sidecar import rollout_from_sidecar  # noqa: E402

_MODEL_NAME = os.environ.get("GRZ_RL_MODEL_NAME", "qwen3.5-4b")

# ---- bench port allocation (concurrent run_agent calls in one actor) ----
_PORT_LOCK = threading.Lock()
_PORTS_IN_USE: set[int] = set()


def _alloc_port(lo=8900, hi=9400) -> int:
    with _PORT_LOCK:
        for _ in range(hi - lo):
            p = random.randint(lo, hi - 1)
            if p in _PORTS_IN_USE:
                continue
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                s.close()
                continue
            s.close()
            _PORTS_IN_USE.add(p)
            return p
        raise RuntimeError(f"no free bench port in [{lo}, {hi})")


def _free_port(p: int):
    with _PORT_LOCK:
        _PORTS_IN_USE.discard(p)


def _parse_spec(prompt: str) -> dict:
    """'chronic=0|horizon=24|seed=0' -> dict (tolerant of extra fields)."""
    spec = {"chronic": 0, "horizon": 24, "seed": 0}
    for part in str(prompt).split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            if k in spec:
                try:
                    spec[k] = int(v)
                except ValueError:
                    pass
    return spec


def _apply_chat_template(tokenizer, messages, tools=None, add_generation_prompt=True):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                tools=tools or None)
        except TypeError:  # template without a tools kwarg
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return "\n".join(f"{m.get('role', '').capitalize()}: {m.get('content', '')}" for m in messages) + "\n"


# ---- Qwen3 completion parsing (mirrors vLLM --reasoning-parser qwen3 --tool-call-parser qwen3_xml) ----
_LT, _GT = "<", ">"
_THINK_OPEN = _LT + "think" + _GT
_THINK_CLOSE = _LT + "/think" + _GT
_TC_OPEN = _LT + "function_call" + _GT
_TC_CLOSE = _LT + "/function_call" + _GT
_PARAM_OPEN = _LT + "parameter="
_PARAM_CLOSE = _LT + "/parameter" + _GT
_THINK_RE = re.compile(re.escape(_THINK_OPEN) + r"(.*?)" + re.escape(_THINK_CLOSE), re.DOTALL)
_TC_RE = re.compile(re.escape(_TC_OPEN) + r"(.*?)" + re.escape(_TC_CLOSE), re.DOTALL)
_PARAM_RE = re.compile(re.escape(_PARAM_OPEN) + r"([^>]+)" + _GT + r"\s*\n(.*?)\n?\s*" + re.escape(_PARAM_CLOSE), re.DOTALL)


def _parse_args(raw_params: str) -> str:
    d = {}
    for m in _PARAM_RE.finditer(raw_params):
        name = m.group(1).strip()
        val = m.group(2).strip()
        try:
            d[name] = json.loads(val)
        except Exception:
            d[name] = val
    return json.dumps(d)


def parse_completion(text: str):
    """raw Qwen3 completion -> (reasoning, content, tool_calls)."""
    m = _THINK_RE.search(text)
    if m:
        reasoning = m.group(1).strip() or None
        rest = text[:m.start()] + text[m.end():]
    else:
        m2 = _TC_RE.search(text)
        idx = text.find(_THINK_OPEN)
        if idx != -1 and (m2 is None or idx < m2.start()):
            end_think = m2.start() if m2 else len(text)
            reasoning = text[idx + len(_THINK_OPEN):end_think].strip() or None
            rest = text[:idx] + text[end_think:]
        else:
            reasoning = None
            rest = text
    tool_calls = []

    def _sub(m):
        inner = m.group(1)
        fnm = re.search(_LT + "function" + _GT + r"\s*\n(.*?)\n?\s*" + _LT + "/function" + _GT,
                        inner, re.DOTALL)
        name = fnm.group(1).strip() if fnm else "unknown"
        tool_calls.append({"id": f"chatcmpl-tool-{uuid4().hex[:16]}", "type": "function",
                           "function": {"name": name, "arguments": _parse_args(inner)}})
        return ""

    rest = _TC_RE.sub(_sub, rest)
    content = rest.strip() or None
    return reasoning, content, tool_calls


def _sse_chunks(cid, model, reasoning, content, tool_calls, finish,
                prompt_ids=None, gen_ids=None, gen_logprobs=None, tokenizer=None):
    """SSE stream in vLLM's return_token_ids shape: prompt_token_ids top-level on
    the first chunk; completion token_ids + per-token logprobs on a choice chunk.
    The gridtrace plugin (opencode side) captures them from this stream."""
    base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
    chunks = []

    def _c(delta, finish_reason=None, choice_extra=None, top=None):
        ch = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        if choice_extra:
            ch.update(choice_extra)
        ev = dict(base)
        ev["choices"] = [ch]
        if top:
            ev.update(top)
        chunks.append("data: " + json.dumps(ev) + "\n\n")

    _c({"role": "assistant", "content": ""},
       top={"prompt_token_ids": prompt_ids} if prompt_ids is not None else None)
    if reasoning:
        _c({"reasoning": reasoning})
    if content:
        _c({"content": content})
    for i, t in enumerate(tool_calls):
        _c({"tool_calls": [{"index": i, "id": t["id"], "type": "function",
                            "function": {"name": t["function"]["name"], "arguments": ""}}]})
        _c({"tool_calls": [{"index": i, "function": {"arguments": t["function"]["arguments"]}}]})
    if gen_ids is not None:
        choice_extra = {"token_ids": list(gen_ids)}
        if gen_logprobs is not None:
            choice_extra["logprobs"] = {"content": [
                {"token": (tokenizer.decode([tid]) if tokenizer else ""),
                 "logprob": lp, "top_logprobs": []} for tid, lp in zip(gen_ids, gen_logprobs)]}
        _c({}, choice_extra=choice_extra)
    _c({}, finish_reason=finish)
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


class AgentExecutor(AgentExecutorBase):
    """OpenAI-compatible server (token-tracing) + GridZero bench as the agent."""

    def _init_server(self, llm_engine, hf_tokenizer):
        self.host = "0.0.0.0"  # reachable from episode netns via the gateway IP
        self.port = self._find_open_port()
        self.llm_engine = llm_engine
        self.hf_tokenizer = hf_tokenizer
        self.model_name = getattr(hf_tokenizer, "name_or_path", "policy-model")
        self._start_server()
        self.client = AsyncOpenAI(base_url=f"http://{self.host}:{self.port}/v1", api_key="EMPTY")
        logger.info(f"GridZero agent server ready at http://{self.host}:{self.port}/v1 (model={self.model_name})")

    @staticmethod
    def _find_open_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _start_server(self):
        app = FastAPI()
        executor = self

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": executor.model_name}

        @app.get("/v1/models")
        async def list_models():
            return {"object": "list", "data": [{"id": executor.model_name, "object": "model", "owned_by": "openrlhf"}]}

        @app.post("/tokenize")
        async def tokenize(request: Request):
            body = await request.json()
            text = _apply_chat_template(executor.hf_tokenizer, body.get("messages", []),
                                        tools=body.get("tools"), add_generation_prompt=False)
            token_ids = executor.hf_tokenizer.encode(text)
            return {"tokens": token_ids, "count": len(token_ids), "max_model_len": executor.max_length}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            messages = body.get("messages", [])
            if not messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            # Token capture happens on the opencode side (the gridtrace plugin
            # tees this stream), so the server only serves: render, generate,
            # parse, stream — with token ids + logprobs in the SSE for the plugin.
            max_tokens = body.get("max_tokens", executor.sampling_params.max_tokens)
            temperature = body.get("temperature", executor.sampling_params.temperature)
            top_p = body.get("top_p", executor.sampling_params.top_p)
            tools = body.get("tools")

            prompt_text = _apply_chat_template(executor.hf_tokenizer, messages,
                                               tools=tools, add_generation_prompt=True)
            prompt_token_ids = executor.hf_tokenizer.encode(prompt_text, add_special_tokens=False)

            remaining = executor.max_length - len(prompt_token_ids)
            if remaining <= 0:
                raise HTTPException(status_code=400,
                                    detail=f"Prompt ({len(prompt_token_ids)}) exceeds max_length ({executor.max_length})")

            sp = SamplingParams(max_tokens=min(max_tokens, remaining), temperature=temperature,
                                top_p=top_p, logprobs=1)
            output = await executor.llm_engine.generate(prompt_token_ids, sp, request_id=uuid4().hex)
            gen = output.outputs[0]
            gen_ids = list(gen.token_ids)
            gen_logprobs = None
            if gen.logprobs:
                gen_logprobs = [lp.get(tid).logprob if lp.get(tid) is not None else 0.0
                                for tid, lp in zip(gen_ids, gen.logprobs)]

            reasoning, content, tool_calls = parse_completion(gen.text or "")
            finish = "tool_calls" if tool_calls else (gen.finish_reason or "stop")
            cid = f"chatcmpl-{uuid4().hex[:24]}"
            model_id = body.get("model", executor.model_name)
            usage = {"prompt_tokens": len(prompt_token_ids), "completion_tokens": len(gen_ids),
                     "total_tokens": len(prompt_token_ids) + len(gen_ids)}

            if body.get("stream"):
                return Response(content=_sse_chunks(cid, model_id, reasoning, content, tool_calls, finish,
                                                    prompt_ids=prompt_token_ids, gen_ids=gen_ids,
                                                    gen_logprobs=gen_logprobs, tokenizer=executor.hf_tokenizer),
                                media_type="text/event-stream")
            message = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning"] = reasoning
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model_id,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage}

        thread = threading.Thread(
            target=lambda: uvicorn.run(app, host=executor.host, port=executor.port, log_level="info", loop="asyncio"),
            daemon=True)
        thread.start()
        for _ in range(60):
            try:
                urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("GridZero agent server failed to start within 60s")

    async def run_agent(self, prompt: str, label: str):
        """One episode: real bench run with opencode pointed at this server.

        The gridtrace plugin inside opencode (enabled via GRZ_RL_TRACE) tees
        every request/response into a per-port sidecar with authoritative token
        ids + logprobs; the returned trace_dir points at it."""
        spec = _parse_spec(prompt)
        port = _alloc_port()
        outdir = os.path.join(_GZ, "runs", f"rl-ep-{port}-{uuid4().hex[:6]}")
        trace_dir = os.path.join(_GZ, ".sb", f"trace-{port + 200}")
        old_trace = os.environ.get("GRZ_RL_TRACE")
        os.environ["GRZ_RL_TRACE"] = "1"
        try:
            from bench.netns import EpisodeNet
            net = EpisodeNet(port, log_dir=os.path.join(_GZ, "runs"))
            net.setup()
            # server binds 0.0.0.0 on the host; from the episode netns it is
            # reachable via the gateway IP (no forwarder hop needed)
            model_url = f"http://{net.gw}:{self.port}/v1"

            a = argparse.Namespace(
                chronic=spec["chronic"], horizon=spec["horizon"], seed=spec["seed"],
                model=_MODEL_NAME, cfg=run_llm.load_config("baseline"),
                attacks=[], adversarial=False, dn_anchor=None,
                safety_cap_h=1.0, liveness_min=5.0, stall_min=10.0, poke_idle_s=60)
            os.makedirs(outdir, exist_ok=True)
            res = await asyncio.to_thread(run_llm.run_episode, port, a, outdir, spec["horizon"],
                                          net=net, model_url=model_url)
            if res is None or res.agent_failed:
                logger.warning(f"episode c{spec['chronic']}@{spec['horizon']} agent_failed: "
                               f"{'backend did not boot' if res is None else res.notes}")
                return {"reward": 0.0, "scores": 0.0, "trace_dir": trace_dir,
                        "extra_logs": {"agent_failed": True,
                                       "notes": "backend_failed" if res is None else res.notes}}
            return {
                "reward": float(res.cum_reward),
                "scores": res.survived / max(1, res.horizon),
                "trace_dir": trace_dir,
                "extra_logs": {"survived": res.survived, "n_trips": res.n_trips,
                               "n_illegal": res.n_illegal, "game_over": res.game_over,
                               "wall_s": res.wall_clock_s, "tok_in": res.tokens_in},
            }
        finally:
            _free_port(port)
            if old_trace is None:
                os.environ.pop("GRZ_RL_TRACE", None)
            else:
                os.environ["GRZ_RL_TRACE"] = old_trace

    async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None):
        self.sampling_params = sampling_params
        self.max_length = max_length
        if not hasattr(self, "client"):
            self._init_server(llm_engine, hf_tokenizer)

        result = await self.run_agent(prompt, label)
        row = rollout_from_sidecar(result["trace_dir"])
        if row is None:
            logger.warning(f"no sidecar turns in {result['trace_dir']}; emitting empty row")
            pids = hf_tokenizer.encode(prompt, add_special_tokens=False)
            row = {"observation_tokens": list(pids), "action_ranges": [],
                   "rollout_log_probs": None, "truncated": False}
        row["prompt"] = prompt
        row["label"] = label
        row["images"] = images
        row["mm_train_inputs"] = None
        row["reward"] = result["reward"]
        row["scores"] = result["scores"]
        row["extra_logs"] = result.get("extra_logs", {})
        return row
