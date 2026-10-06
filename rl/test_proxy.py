"""T0 — proxy unit test: field injection + SSE byte-fidelity + passthrough.

Run with the OpenRLHF venv (has aiohttp):
  ~/Documents/openrlhf/.venv/bin/python ~/Documents/GridZero/rl/test_proxy.py
"""
import asyncio, json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy import Proxy
from aiohttp import web, ClientSession

SSE_BODY = (b'data: {"id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}\n\n'
            b'data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "hello grid"}, "finish_reason": null}]}\n\n'
            b'data: {"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'
            b'data: [DONE]\n\n')

received = []


async def fake_handler(request):
    if request.method == "POST" and request.path.endswith("/chat/completions"):
        body = json.loads(await request.read())
        received.append(body)
        return web.Response(body=SSE_BODY, content_type="text/event-stream")
    if request.path.endswith("/models"):
        return web.json_response({"object": "list", "data": [{"id": "policy", "object": "model", "owned_by": "fake"}]})
    return web.json_response({"echo": request.path, "method": request.method})


async def main():
    tmp = tempfile.mkdtemp(prefix="t0proxy")
    # fake agent server
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", fake_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    target_port = site._server.sockets[0].getsockname()[1]
    target = f"http://127.0.0.1:{target_port}/v1"

    px = Proxy(target, session_id="test-sess-123", host="127.0.0.1", tee_dir=tmp)
    url = px.start()

    failures = []
    async with ClientSession() as cs:
        # 1. chat completion: injection + SSE fidelity
        payload = {"model": "policy", "stream": True, "max_tokens": 8,
                   "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "bash"}}]}
        async with cs.post(url + "/chat/completions", json=payload) as r:
            data = await r.read()
        if data != SSE_BODY:
            failures.append(f"SSE mismatch: {data[:80]!r} != {SSE_BODY[:80]!r}")
        if not received:
            failures.append("target got no request")
        else:
            got = received[0]
            if got.get("session_id") != "test-sess-123":
                failures.append(f"session_id not injected: {got.get('session_id')}")
            if got.get("logprobs") is not True:
                failures.append("logprobs not injected")
            if got.get("top_logprobs") != 1:
                failures.append("top_logprobs not injected")
            if got.get("messages") != payload["messages"]:
                failures.append("messages mangled")
            if got.get("tools") != payload["tools"]:
                failures.append("tools mangled")
            if got.get("max_tokens") != 8:
                failures.append("max_tokens mangled")

        # 2. GET passthrough
        async with cs.get(url + "/models") as r:
            j = await r.json()
        if j["data"][0]["id"] != "policy":
            failures.append(f"models passthrough failed: {j}")

        # 3. odd path passthrough
        async with cs.post(url + "/tokenize", json={"messages": []}) as r:
            j = await r.json()
        if j.get("echo") != "/v1/tokenize":  # target base ends in /v1
            failures.append(f"generic passthrough failed: {j}")

    px.stop()
    await runner.cleanup()

    # 4. tee file
    tee = os.path.join(tmp, "proxy.jsonl")
    if not os.path.exists(tee):
        failures.append("tee file missing")
    else:
        rec = json.loads(open(tee).readline())
        if rec["body"].get("session_id") != "test-sess-123":
            failures.append("tee record missing session_id")
        if not str(rec["body"].get("messages", "")).startswith("<"):
            failures.append(f"tee messages not slimmed: {rec['body'].get('messages')}")

    if failures:
        print("T0 FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("T0 PASS: injection, SSE byte-fidelity, passthrough, tee all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
