"""opencode_driver.py — drives `opencode serve` (C5) and translates its raw SSE
into C4 agent.* events on the bus. Degrades gracefully: if opencode isn't
available, the sim + SSE + controls still work (a system event is emitted).

opencode routes used (validated against v1.18.34):
  POST /api/session                         create session
  POST /api/session/{id}/prompt             blocking prompt (kickoff)
  POST /api/session/{id}/prompt_async       non-blocking steer/inject
  POST /api/session/{id}/abort              pause
  GET  /event                               raw agent event stream (SSE)
"""
from __future__ import annotations
import asyncio, json, os, re, subprocess, time
import httpx

DEFAULT_PROVIDER = os.environ.get("OPENCODE_PROVIDER", "vllm4b")

class OpenCodeDriver:
    def __init__(self, bus, root, sim_port=8731, oc_port=None, attacker=False, atk_token=""):
        self.bus = bus
        self.root = root
        # attacker token: lets `simctl attack` pass the backend guard for the
        # ATTACKER session only (the defender's simctl never sees it).
        self.atk_token = atk_token
        # agent-vs-agent: when True this drives the ATTACKER session — a separate
        # opencode + sandbox sharing the SAME sim (sim_port). Its events are tagged
        # "attacker.*" (not "agent.*") so the defender's trace stays clean, and it
        # never emits the shared session.status STATE frame (that's the UI's agent).
        self.attacker = attacker
        self.tag = "attacker" if attacker else "agent"
        # Per-backend port isolation: each backend gives ITS opencode session the
        # correct SIM_API_URL (its own port) and runs opencode on a distinct port,
        # so multiple episodes can run in parallel without cross-contamination.
        self.sim_port = sim_port
        self.oc_port = oc_port or (sim_port + 200)
        # Sandbox (benchmark validity): when OPENCODE_SANDBOX=1 the model runs
        # inside a bubblewrap FS that contains ONLY the operator workspace, so it
        # cannot read bench/ (metric+panel), runs/ (baselines), backend/, etc.
        # Off by default (interactive harness keeps the full workspace).
        self.sandbox = os.environ.get("OPENCODE_SANDBOX", "0") == "1"
        self.with_docs = os.environ.get("OPENCODE_NO_DOCS", "0") != "1"
        self.doc_warning = os.environ.get("OPENCODE_DOC_WARNING", "")  # A/B doc variant
        self.model = os.environ.get("OPENCODE_MODEL", "qwen3.5-4b")  # cross-model
        self.base = f"http://127.0.0.1:{self.oc_port}"
        self.session_id = None
        self.model = os.environ.get("OPENCODE_MODEL", "qwen3.5-4b")
        self.variant = os.environ.get("OPENCODE_VARIANT", "")
        self._proc = None
        self._task = None
        self._turn_id = None
        self._turn_start = 0.0
        self._tool_started = {}   # part_id -> (t0, tool)
        self.available = False
        self._client = None
        self.n_agent_events = 0   # count of agent.* events consumed (liveness signal)
        self.last_model_output_ts = time.time()  # real model output (delta/tool), for stall watchdog
        self._kickoff_text = ""  # stored so restart_agent can re-kick the same task

    # ---- lifecycle ----
    async def start(self):
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base, timeout=60)
        # capture opencode stderr to a per-port log (was DEVNULL; needed to diagnose
        # slow/failed sandbox startup under concurrent load).
        logpath = os.path.join(self.root, "runs", f"opencode-{self.oc_port}.log")
        try:
            logf = open(logpath, "ab")
        except PermissionError:
            # a previous root (netns) backend owns this file and this one isn't root
            subprocess.run(["sudo", "-n", "chown", f"{os.getuid()}:", logpath], capture_output=True)
            logf = open(logpath, "ab")
        try:
            if self.sandbox:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                from bench import sandbox
                argv, _meta = sandbox.setup(self.oc_port, self.sim_port,
                                            with_docs=self.with_docs,
                                            doc_warning=self.doc_warning, model=self.model,
                                            attacker=self.attacker, atk_token=self.atk_token,
                                            vllm_url=os.environ.get("VLLM_FORWARD_URL", "") or None)
                self._proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT)
            else:
                self._proc = subprocess.Popen(
                    ["opencode", "serve", "--hostname", "127.0.0.1", "--port", str(self.oc_port)],
                    cwd=self.root,
                    env={**os.environ,
                         "SIM_API_URL": f"http://127.0.0.1:{self.sim_port}",
                         "PATH": os.path.join(self.root, "cli") + os.pathsep + os.environ.get("PATH", ""),
                         # anti reward-hacking: the model can never reset the episode,
                         # never holds the operator token; only the attacker gets the
                         # attack token (and SIMCTL_ATTACKER to unlock that subcommand).
                         "SIMCTL_NO_RESET": "1",
                         "SIM_API_TOKEN": self.atk_token if self.attacker else "",
                         **({"SIMCTL_ATTACKER": "1"} if self.attacker else {})},
                    stdout=logf, stderr=subprocess.STDOUT)
            # Sandboxed opencode (bwrap + node + model init) under concurrent load
            # can take well over 15s to become healthy — wait up to 180s with retries.
            ok = False
            for attempt in range(180):
                if await self._health():
                    ok = True
                    break
                # if our process already died, don't keep waiting on it
                if self._proc.poll() is not None and attempt > 10:
                    break
                await asyncio.sleep(1.0)
            # Guard: if our `opencode serve` process died but the port still answers
            # health, a STALE server owns that port — do not attach to it (parallel
            # safety). Fail the agent rather than drive the wrong sim.
            if ok and self._proc.poll() is not None:
                self._emit_system("error", f"opencode port {self.oc_port} owned by a stale server; "
                                           f"refusing to attach. Kill it and retry.")
                ok = False
            if ok:
                self.available = True
                self._task = asyncio.create_task(self._consume())
                self._emit_system("info", f"opencode ready (sandbox={self.sandbox}, port {self.oc_port})")
            else:
                self._emit_system("error", "opencode serve not reachable after 180s; agent UNAVAILABLE "
                                           "(this episode must be marked agent_failed, not run without an agent)")
        except Exception as e:
            self._emit_system("error", f"opencode unavailable: {e}")
        finally:
            try:
                logf.flush()
            except Exception:
                pass

    def active(self) -> bool:
        """True iff the agent is usable: opencode up AND a session was created."""
        return bool(self.available and self.session_id)

    async def stop(self):
        try:
            if self._task:
                self._task.cancel()
            if self._proc:
                if self.sandbox:
                    # sandboxed opencode runs as ROOT under sudo bwrap — a plain
                    # terminate() from our non-root process is EPERM. Kill via sudo;
                    # also kill the opencode INSIDE the namespace (by its unique
                    # port) since killing the bwrap wrapper orphans it at 100% CPU.
                    import sys
                    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    from bench import sandbox
                    await asyncio.to_thread(sandbox.kill_sandbox_proc, self._proc, self.oc_port)
                else:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
            if self._client:
                await self._client.aclose()
        except Exception:
            pass

    async def _health(self):
        try:
            r = await self._client.get("/global/health")
            return r.status_code == 200
        except Exception:
            return False

    # ---- session ----
    @staticmethod
    def _model_ref(model, provider, variant):
        """opencode model ref: {id, providerID, variant?}. 'default'/empty variant
        is omitted (opencode then uses the model's default reasoning)."""
        ref = {"id": model, "providerID": provider}
        if variant and variant != "default":
            ref["variant"] = variant
        return ref

    async def create_session(self, agent="build", provider=DEFAULT_PROVIDER, model="qwen3.5-4b", variant=None):
        if not self.available:
            return None
        try:
            # NOTE: opencode's model ref uses the key "id" (NOT "modelID").
            r = await self._client.post("/api/session",
                                        json={"agent": agent,
                                              "model": self._model_ref(model, provider, variant)})
            self.session_id = r.json().get("data", {}).get("id")
            self.model, self.variant = model, variant
            return self.session_id
        except Exception as e:
            self._emit_system("error", f"create_session failed: {e}")
            return None

    async def set_model(self, provider=DEFAULT_PROVIDER, model=None, variant=None):
        """Live-switch the active session's model/variant (best-effort). Body is
        {"model": {id, providerID, variant?}} (verified -> 204)."""
        if not self.available or not self.session_id:
            return False
        model = model or self.model
        variant = variant if variant is not None else self.variant
        try:
            r = await self._client.post(f"/api/session/{self.session_id}/model",
                                        json={"model": self._model_ref(model, provider, variant)})
            if r.status_code in (200, 204):
                self.model, self.variant = model, variant
                self._emit_system("info", f"model -> {model}" + (f" ({variant})" if variant and variant != "default" else ""))
                return True
            self._emit_system("warn", f"set_model http {r.status_code}")
            return False
        except Exception as e:
            self._emit_system("warn", f"set_model failed: {e}")
            return False

    async def list_models(self, provider=DEFAULT_PROVIDER):
        """Available models + reasoning variants for the UI selector.

        Read from the opencode.json config (the ground truth the gateway supports);
        the live /config/providers card is used as a secondary merge if it happens
        to expose variants (it often returns them empty). Returns
        [{id, name, variants:[...]}] where variants includes 'default' first.
        """
        models = {}
        # primary: parse the config file (deterministic)
        cfg = os.path.expanduser("~/.config/opencode/opencode.json")
        try:
            import json as _j
            d = _j.load(open(cfg))
            card = (d.get("provider", {}).get(provider, {}) or {}).get("models", {}) or {}
            for mid, m in card.items():
                variants = list((m.get("variants") or {}).keys()) if isinstance(m, dict) else []
                models[mid] = {"id": mid, "name": m.get("name", mid) if isinstance(m, dict) else mid,
                               "variants": ["default"] + [v for v in variants if v != "default"]}
        except Exception as e:
            self._emit_system("warn", f"list_models config parse failed: {e}")
        # secondary: merge any variants the live card exposes
        if self.available:
            try:
                r = await self._client.get("/config/providers")
                data = r.json()
                provs = data.get("providers", data) if isinstance(data, dict) else data
                target = None
                if isinstance(provs, list):
                    target = next((p for p in provs if isinstance(p, dict) and p.get("id") == provider), None)
                elif isinstance(provs, dict):
                    target = provs.get(provider) or (provs.get("providers", {}) or {}).get(provider)
                for mid, card in ((target or {}).get("models", {}) or {}).items():
                    live_variants = list((card.get("variants") or {}).keys()) if isinstance(card, dict) else []
                    if mid in models:
                        merged = list(dict.fromkeys(models[mid]["variants"] +
                                                     [v for v in live_variants if v != "default"]))
                        models[mid]["variants"] = merged
                    else:
                        models[mid] = {"id": mid, "name": card.get("name", mid) if isinstance(card, dict) else mid,
                                       "variants": ["default"] + [v for v in live_variants if v != "default"]}
            except Exception:
                pass
        return list(models.values())

    async def _fire_prompt_async(self, text):
        # NOTE: the route is /session/{id}/prompt_async (NO /api prefix). The
        # /api/session/{id}/prompt route wants a "prompt" key and 400s with parts.
        r = await self._client.post(f"/session/{self.session_id}/prompt_async",
                                    json={"parts": [{"type": "text", "text": text}]})
        return r.status_code

    async def kickoff(self, text):
        if not self.available or not self.session_id:
            return
        self._kickoff_text = text
        try:
            sc = await self._fire_prompt_async(text)
            if sc not in (200, 202, 204):
                self._emit_system("warn", f"kickoff http {sc}")
        except Exception as e:
            self._emit_system("error", f"kickoff failed: {e}")

    async def restart(self, resume_text=""):
        """Break a wedged agent: abort the current session, create a FRESH one,
        and re-kick. (After a runaway turn the opencode session can stop generating
        even on new prompts — only a fresh session recovers it.)"""
        if not self.available:
            self._emit_system("error", "restart failed: opencode not available")
            return False
        try:
            if self.session_id:
                await self._client.post(f"/session/{self.session_id}/abort")
        except Exception:
            pass
        try:
            sid = await self.create_session()
            if sid:
                self.last_model_output_ts = time.time()
                txt = resume_text or self._kickoff_text
                if txt:
                    await self._fire_prompt_async(txt)
                self._emit_system("info", f"agent restarted (fresh session {sid[:16]}…)")
                return True
            self._emit_system("error", "restart failed: create_session returned None")
            return False
        except Exception as e:
            self._emit_system("error", f"restart failed: {e}")
            return False

    async def steer(self, text):
        if not self.available or not self.session_id:
            self._emit_system("warn", "steer ignored: agent not running")
            return
        try:
            await self._fire_prompt_async(f"[OPERATOR] {text}")
        except Exception as e:
            self._emit_system("error", f"steer failed: {e}")

    async def pause(self):
        if self.available and self.session_id:
            try:
                await self._client.post(f"/session/{self.session_id}/abort")  # no /api
            except Exception as e:
                self._emit_system("warn", f"pause failed: {e}")

    async def session_stats(self) -> dict:
        """Cumulative token/cost usage for the active session (from opencode)."""
        if not self.available or not self.session_id:
            return {"available": False}
        try:
            r = await self._client.get(f"/api/session/{self.session_id}")
            d = r.json().get("data", {})
            tok = d.get("tokens", {}) or {}
            return {"available": True, "session_id": self.session_id,
                    "tokens_in": tok.get("input", 0), "tokens_out": tok.get("output", 0),
                    "reasoning_tokens": tok.get("reasoning", 0),
                    "cache_read": tok.get("cache", {}).get("read", 0),
                    "cache_write": tok.get("cache", {}).get("write", 0),
                    "cost_usd": d.get("cost", 0.0), "title": d.get("title", "")}
        except Exception as e:
            return {"available": True, "session_id": self.session_id, "error": str(e)}

    # ---- event consumption + translation ----
    async def _consume(self):
        try:
            async with httpx.AsyncClient(base_url=self.base, timeout=None) as c:
                async with c.stream("GET", "/event") as resp:
                    buf = ""
                    async for chunk in resp.aiter_text():
                        buf += chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            if line.startswith("data: "):
                                frame = line[6:].strip()
                                if frame:
                                    self._translate(frame)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._emit_system("warn", f"opencode event stream ended: {e}")

    def _translate(self, frame):
        try:
            ev = json.loads(frame)
        except Exception:
            return
        t = ev.get("type")
        p = ev.get("properties", {}) or {}
        # liveness counter: any model-side activity (streaming/tool/session)
        if t in ("message.part.delta", "message.part.updated", "message.updated",
                 "reasoning", "session.status", "session.idle", "idle",
                 "step-start", "step-finish"):
            self.n_agent_events += 1
        if t == "message.part.delta":
            self._ensure_turn()
            delta = p.get("delta", "")
            if delta:
                self.last_model_output_ts = time.time()
            self.bus.emit(f"{self.tag}.delta", {"turn": self._turn_id, "message_id": p.get("messageID"),
                                                "part_id": p.get("partID"), "field": p.get("field", "text"),
                                                "delta": delta})
        elif t == "reasoning":
            self._ensure_turn()
            self.bus.emit(f"{self.tag}.delta", {"turn": self._turn_id, "part_id": p.get("partID"),
                                                "field": "reasoning", "delta": p.get("part", {}).get("text", "") if isinstance(p.get("part"), dict) else ""})
        elif t == "message.part.updated":
            part = p.get("part", {}) or {}
            ptype = part.get("type")
            if ptype == "reasoning":
                self._ensure_turn()
                self.bus.emit(f"{self.tag}.delta", {"turn": self._turn_id, "part_id": part.get("id"),
                                                    "field": "reasoning", "delta": part.get("text", "")})
            elif ptype == "tool":
                self._handle_tool(part)
        elif t == "session.status":
            st = (p.get("status") or {}).get("type", "idle")
            if st == "busy" and self._turn_id is None:
                self._ensure_turn()
            elif st in ("idle",):
                self._turn_end()
            if not self.attacker:
                self.bus.emit("session.status", {"status": st, "title": "grid agent"})
        elif t in ("session.idle", "idle"):
            self._turn_end()
            if not self.attacker:
                self.bus.emit("session.status", {"status": "idle", "title": "grid agent"})
        elif t == "message.updated":
            info = p.get("info", {}) or {}
            if info.get("role") == "assistant":
                self._ensure_turn()

    def _handle_tool(self, part):
        state = part.get("state", {}) or {}
        status = state.get("status", "pending")
        pid = part.get("id")
        tool = part.get("tool", "?")
        inp = state.get("input", {}) or {}
        self._ensure_turn()
        # emit tool_call once per part (first sight) — input may be {} until it fills in
        if pid not in self._tool_started:
            self._tool_started[pid] = time.time()
            self.bus.emit(f"{self.tag}.tool_call", {"turn": self._turn_id, "part_id": pid, "tool": tool,
                                                    "input": inp})
        if status in ("completed", "error"):
            t0 = self._tool_started.pop(pid, time.time())
            dur = int((time.time() - t0) * 1000)
            output = state.get("output", "")
            if isinstance(output, dict):
                output = json.dumps(output)
            # carry the real command in the result (the pending input was often empty)
            self.bus.emit(f"{self.tag}.tool_result", {"turn": self._turn_id, "part_id": pid, "tool": tool,
                                                      "status": status, "input": inp,
                                                      "output": str(output), "duration_ms": dur})

    def _ensure_turn(self):
        if self._turn_id is None:
            self._turn_id = "t-%03d" % int(time.time() % 100000)
            self._turn_start = time.time()

    def _turn_end(self):
        if self._turn_id is not None:
            self.bus.emit(f"{self.tag}.turn_end", {"turn": self._turn_id,
                                                   "latency_ms": int((time.time() - self._turn_start) * 1000)})
            self._turn_id = None

    def _emit_system(self, level, msg):
        self.bus.emit("system", {"level": level, "msg": msg,
                                 "who": "attacker" if self.attacker else "agent"})


class AttackerSession:
    """The ATTACKER's opencode session in agent-vs-agent mode — a second, independent
    `opencode serve` (oc_port = sim_port+201) in its own sandbox, built with
    attacker=True (simctl + AGENTS-ATTACKER.md, NO defender docs) and the attacker
    model, that shares the SAME sim (SIM_API_URL = sim_port) as the defender.

    The two models are BLIND to each other: each sees only the grid state via simctl.
    The attacker driver tags its events "attacker.*" (see OpenCodeDriver.tag) so they
    never mix into the defender's agent.* trace, and it never emits the shared
    session.status STATE frame. No cross-session event sharing is added.
    """

    def __init__(self, bus, root, sim_port, model="qwen3.5-4b", atk_token=""):
        self.bus = bus
        self.root = root
        self.sim_port = sim_port
        self.oc_port = sim_port + 201  # attacker opencode port (defender is +200)
        self.model = model
        self.driver = OpenCodeDriver(bus, root, sim_port=sim_port, oc_port=self.oc_port,
                                     attacker=True, atk_token=atk_token)
        # The attacker always wants its own guide (build_ws ignores with_docs when
        # attacker=True, but keep the driver's flags explicit).
        self.driver.with_docs = True
        self.driver.doc_warning = ""
        self.started = False

    def _ensure_driver(self):
        self.driver.model = self.model
        self.driver.attacker = True
        self.driver.tag = "attacker"

    async def attacker_start(self, model, kickoff_text):
        """Start (or restart) the attacker session: launch its opencode+sandbox if
        not already up, create a fresh session on the attacker model, and kick it
        off. Returns the session id (None on failure)."""
        self.model = model
        self._ensure_driver()
        if not self.driver.available:
            await self.driver.start()
            self.started = True
        # a fresh session per start/restart (a wedged session only recovers on restart)
        if self.driver.session_id:
            await self.driver.pause()
        self.driver.last_model_output_ts = time.time()
        sid = await self.driver.create_session(model=model)
        if sid:
            await self.driver.kickoff(kickoff_text)
            self.bus.emit("system", {"level": "info", "who": "attacker",
                                     "msg": f"attacker session started (model={model}, port {self.oc_port})"})
        else:
            self.bus.emit("system", {"level": "error", "who": "attacker",
                                     "msg": "attacker session FAILED to start"})
        return sid

    async def attacker_steer(self, text):
        await self.driver.steer(text)

    async def attacker_pause(self):
        await self.driver.pause()

    async def attacker_stop(self):
        try:
            await self.driver.stop()
        except Exception:
            pass
        self.started = False

    def attacker_active(self):
        return self.driver.active()

    async def attacker_stats(self):
        llm = await self.driver.session_stats()
        return {"active": self.driver.active(),
                "oc_port": self.oc_port,
                "session_id": self.driver.session_id,
                "events": self.driver.n_agent_events,
                "last_output_ts": self.driver.last_model_output_ts,
                "llm": llm}
