"""Per-episode translation proxy: opencode -> OpenRLHF agent server.

opencode's OpenAI-compatible client cannot send custom body fields, but the
agent server needs `session_id` (per-episode trace key) and `logprobs`
(rollout logprob capture for the async IS-correction path). This proxy sits
between them: it injects those fields, passes everything else through
unchanged, and streams the SSE response back byte-for-byte.

Run one per episode, bound 0.0.0.0 (reachable from the episode's netns via the
gateway IP) or 127.0.0.1 (no netns). No deps beyond aiohttp (in the OpenRLHF
venv).
"""
import asyncio, json, os, socket, threading

from aiohttp import web, ClientSession, ClientTimeout


def _free_port(host="0.0.0.0"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def make_app(target, session_id, tee_dir=None):
    """Build the aiohttp app. `target` = agent-server base URL (no trailing /)."""

    async def forward(request):
        # public URL is http://host:port/v1, target base also ends in /v1 -> strip
        # the public prefix so the target path isn't doubled
        path = request.path
        if path.startswith("/v1"):
            path = path[len("/v1"):] or "/"
        body = await request.read()
        out_body = body
        if request.method == "POST" and path.endswith("/chat/completions"):
            try:
                d = json.loads(body)
                # Only the main agent's requests (tools-bearing) join the traced
                # session; auxiliary calls (opencode title gen etc.) pass through
                # sessionless so they don't pollute the episode's token trace.
                if d.get("tools"):
                    d["session_id"] = session_id
                    d.setdefault("logprobs", True)
                    d.setdefault("top_logprobs", 1)
                out_body = json.dumps(d).encode()
                if tee_dir:
                    os.makedirs(tee_dir, exist_ok=True)
                    if os.environ.get("GRZ_PROXY_DEBUG"):
                        rec = {"path": request.path, "body": d}
                    else:
                        rec = {"path": request.path,
                               "body": {k: (f"<{len(v)} msgs>" if k == "messages" else v) for k, v in d.items()}}
                    with open(os.path.join(tee_dir, "proxy.jsonl"), "a") as f:
                        f.write(json.dumps(rec) + "\n")
            except Exception as e:
                print(f"  (proxy: body injection failed: {e})")

        hdrs = {"Content-Type": "application/json"}
        async with ClientSession() as cs:
            async with cs.request(request.method, target + path,
                                  data=out_body, headers=hdrs,
                                  timeout=ClientTimeout(total=None, sock_read=900)) as up:
                resp = web.StreamResponse(status=up.status,
                                          headers={k: v for k, v in up.headers.items()
                                                   if k.lower() not in ("content-length", "transfer-encoding")})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", forward)
    return app


class Proxy:
    """Threaded aiohttp proxy with a start()/stop() lifecycle."""

    def __init__(self, target, session_id, port=None, host="0.0.0.0", tee_dir=None):
        self.target = target.rstrip("/")
        self.session_id = session_id
        self.port = port or _free_port(host)
        self.host = host
        self.tee_dir = tee_dir
        self._loop = None
        self._thread = None

    def start(self):
        ready = threading.Event()

        def _serve():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            runner = web.AppRunner(make_app(self.target, self.session_id, self.tee_dir))
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, self.host, self.port).start())
            self._loop = loop
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        if not ready.wait(15):
            raise RuntimeError("proxy did not start within 15s")
        return self.url

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/v1"

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
