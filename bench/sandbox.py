"""bench/sandbox.py — filesystem sandbox for benchmark LLM episodes (bubblewrap).

WHY: in the pilot, the model (run with opencode cwd = the repo root) read the
entire benchmark "answer key" — bench/score.py (the metric), bench/panel.json
(the chronic ids), runs/ (DoNothing baseline results), the reward source, and
PLAN-BENCH.md — via absolute paths. That invalidates the measurement (e.g. an
episode that matched do-nothing exactly after reading the DN score). The fix: run
opencode inside a bubblewrap sandbox whose filesystem contains ONLY the operator
workspace (simctl + docs + renders). The real repo's benchmark internals simply
do not exist in the sandbox's filesystem.

Validated (2026-10-03): inside the sandbox `ls /home/nate/grid2op-harness` shows
only AGENTS.md/cli/docs/recipes/render; `cat bench/score.py`, `ls bench/`, and
`ls runs/` all return "No such file or directory"; `simctl` runs.

Isolation details:
- The clean operator workspace (a COPY, not symlinks — bwrap can't follow
  out-of-sandbox symlink targets) is bind-mounted at the real repo path
  /home/nate/grid2op-harness, so the model's cwd looks normal.
- Each sandbox gets its OWN per-port HOME (a fresh copy of the opencode config +
  gateway auth) so PARALLEL sessions don't collide on opencode's data dir.
- /home/nate/.nvm (opencode binary + node) is shared read-only (safe).
- The render dir is bind-mounted read-only at its real path so `simctl render`
  PNGs (written by the host backend) are readable by the model.
- opencode runs as ROOT under `sudo bwrap` (user namespaces are disabled here),
  so callers must terminate it with `sudo kill`, not a plain signal.
"""
from __future__ import annotations
import os, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = ROOT                                   # /home/nate/grid2op-harness
REPO_NAME = os.path.basename(ROOT)                  # grid2op-harness
SANDBOX_ROOT = os.path.join(ROOT, ".sb")            # per-port homes (gitignored)
WS_ROOT = os.path.join(ROOT, ".sandbox-ws")         # clean operator workspace (gitignored)
HOME = "/home/nate"
HOME_IN_SB = "/home/nate"

# system dirs needed for python3 + node + opencode to run
_SYS_BINDS = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"]


def _opencode_bin():
    b = shutil.which("opencode")
    if not b:
        raise RuntimeError("opencode not found on PATH")
    return os.path.realpath(b)


def _opencode_runtime_binds():
    """(ro-binds, node_bin_dir) for the opencode runtime, layout-agnostic.
    Handles both an nvm install (opencode+node under ~/.nvm) and a standalone
    binary (e.g. ~/.opencode/bin) with system node in /usr/bin."""
    binds = []
    node_bin = "/usr/bin"
    nvm = os.path.join(HOME, ".nvm")
    if os.path.isdir(nvm):
        binds.append((nvm, nvm))
        vroot = os.path.join(nvm, "versions", "node")
        if os.path.isdir(vroot):
            versions = sorted(os.listdir(vroot))
            if versions:
                cand = os.path.join(vroot, versions[-1], "bin")
                if os.path.isdir(cand):
                    node_bin = cand
    oc_dir = os.path.dirname(_opencode_bin())
    binds.append((oc_dir, oc_dir))  # overlays the per-port HOME bind
    return binds, node_bin


def build_ws(with_docs=True, oc_port=None, doc_warning=None, attacker=None):
    """Build the clean operator workspace. Returns its path.

    Per-port workspace (oc_port) so PARALLEL episodes each own their copy — a shared
    path that gets rmtree'd on every call would wipe a sibling's workspace mid-run.
    `doc_warning` (A/B): optional text appended to the copied AGENTS.md, so a
    variant config (e.g. the loss-of-load warning) differs from the baseline ONLY
    by that doc — everything else (simctl, docs, recipes) is identical.
    `attacker` (agent-vs-agent): build the ATTACKER workspace instead — cli/ (simctl,
    which has the `attack` command) + the attacker guide (AGENTS-ATTACKER.md, renamed
    to AGENTS.md so opencode auto-loads it) and NO defender docs/recipes (the attacker
    must not get the defender's strategy guide).
    """
    ws = WS_ROOT if oc_port is None else os.path.join(WS_ROOT + f"-{oc_port}")
    if os.path.exists(ws):
        shutil.rmtree(ws)
    os.makedirs(ws, exist_ok=True)
    # simctl (the model's only tool) — copy, drop tests/pycache
    dst_cli = os.path.join(ws, "cli")
    shutil.copytree(os.path.join(ROOT, "cli"), dst_cli,
                    ignore=shutil.ignore_patterns("tests", "__pycache__"))
    if not os.access(os.path.join(dst_cli, "simctl"), os.X_OK):
        os.chmod(os.path.join(dst_cli, "simctl"), 0o755)
    if attacker:
        guide_src = os.path.join(ROOT, "AGENTS-ATTACKER.md")
        if not os.path.isfile(guide_src):
            raise RuntimeError(f"attacker workspace: {guide_src} missing")
        shutil.copy(guide_src, os.path.join(ws, "AGENTS.md"))
        return ws
    if with_docs:
        for item in ("AGENTS.md", "docs", "recipes"):
            src = os.path.join(ROOT, item)
            dst = os.path.join(ws, item)
            if os.path.isfile(src):
                shutil.copy(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
        if doc_warning and os.path.isfile(os.path.join(ws, "AGENTS.md")):
            with open(os.path.join(ws, "AGENTS.md"), "a") as f:
                f.write("\n\n" + doc_warning.rstrip() + "\n")
    return ws


def build_sandbox_home(oc_port, model=None):
    """Per-port HOME with a private copy of opencode config + gateway auth, so
    parallel sandboxes don't collide on opencode's data dir.

    `model` (cross-model): rewrite the copied opencode.json so the build/plan
    agents + global default use `<provider>/<model>` (OPENCODE_PROVIDER, default vllm4b). Verified: opencode's session
    model comes from the config's agent.model pin, NOT from a session-level
    override, so this is the reliable way to switch models per episode.
    """
    home = os.path.join(SANDBOX_ROOT, f"home-{oc_port}")
    os.makedirs(os.path.join(home, ".config", "opencode"), exist_ok=True)
    os.makedirs(os.path.join(home, ".local", "share", "opencode"), exist_ok=True)
    os.makedirs(os.path.join(home, ".cache", "opencode"), exist_ok=True)
    # copy the provider/model config
    cfg_src = os.path.join(HOME, ".config/opencode/opencode.json")
    cfg_dst = os.path.join(home, ".config/opencode/opencode.json")
    if os.path.exists(cfg_src):
        shutil.copy(cfg_src, cfg_dst)
        if model:
            _patch_config_model(cfg_dst, model)
    # copy the gateway auth (needed for the model to reach the LLM)
    auth_src = os.path.join(HOME, ".local/share/opencode/auth.json")
    if os.path.exists(auth_src):
        shutil.copy(auth_src, os.path.join(home, ".local/share/opencode/auth.json"))
    return home


def _patch_config_model(cfg_path, model):
    """Set global model + agent.build.model + agent.plan.model to {provider}/{model}."""
    import json
    provider = os.environ.get("OPENCODE_PROVIDER", "vllm4b")
    full = model if "/" in model else f"{provider}/{model}"
    try:
        d = json.load(open(cfg_path))
        d["model"] = full
        d.setdefault("agent", {})
        for agent_key in ("build", "plan"):
            d["agent"].setdefault(agent_key, {})["model"] = full
        json.dump(d, open(cfg_path, "w"), indent=2)
    except Exception as e:
        print(f"  (warn) could not patch sandbox model config: {e}")


def bwrap_argv(oc_port, sim_port, ws, sb_home, attacker=False, atk_token=""):
    """Full argv to launch a sandboxed `opencode serve` (runs as root via sudo)."""
    oc_bin = _opencode_bin()
    runtime_binds, node_bin = _opencode_runtime_binds()
    real_render = os.path.join(ROOT, "render")
    args = ["sudo", "-n", "bwrap"]
    for d in _SYS_BINDS:
        args += ["--ro-bind", d, d]
    # per-sandbox HOME (rw) = config + auth + cache, isolated per port. Bind this
    # FIRST, then layer the shared runtime + workspace on top (bwrap applies binds
    # in order; a later bind to a deeper path overlays an earlier broader one).
    args += ["--bind", sb_home, HOME_IN_SB]
    # shared read-only opencode runtime (node + opencode binary), overlaid under HOME
    for src, dst in runtime_binds:
        args += ["--ro-bind", src, dst]
    # the operator workspace AT the repo path (rw copy)
    args += ["--bind", ws, REPO_PATH]
    # render PNGs (written by the host backend) readable at their real path
    if os.path.isdir(real_render):
        args += ["--ro-bind", real_render, os.path.join(REPO_PATH, "render")]
    # ephemeral + namespace isolation
    args += ["--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc",
             "--unshare-pid", "--unshare-uts"]
    args += ["--chdir", REPO_PATH]
    args += ["--setenv", "HOME", HOME_IN_SB]
    args += ["--setenv", "PATH", f"{node_bin}:{REPO_PATH}/cli:/usr/bin:/bin"]
    args += ["--setenv", "SIM_API_URL", f"http://127.0.0.1:{sim_port}"]
    # anti reward-hacking: the model can never reset the episode and never holds
    # the operator token; only the attacker gets the attack token + subcommand.
    args += ["--setenv", "SIMCTL_NO_RESET", "1"]
    args += ["--setenv", "SIM_API_TOKEN", atk_token if attacker else ""]
    if attacker:
        args += ["--setenv", "SIMCTL_ATTACKER", "1"]
    args += ["--", oc_bin, "serve", "--hostname", "127.0.0.1", "--port", str(oc_port)]
    return args


def setup(oc_port, sim_port, with_docs=True, doc_warning=None, model=None, attacker=None,
          atk_token=""):
    """Build per-port ws + sandbox home; return (argv, {ws, sb_home})."""
    ws = build_ws(with_docs, oc_port=oc_port, doc_warning=doc_warning, attacker=attacker)
    sb_home = build_sandbox_home(oc_port, model=model)
    return bwrap_argv(oc_port, sim_port, ws, sb_home, attacker=attacker, atk_token=atk_token), \
        {"ws": ws, "sb_home": sb_home}


def _root_pids_with_port(oc_port):
    """Host PIDs of the whole sandbox for `oc_port`: the bwrap wrappers AND the
    namespaced opencode (the inner binary is `opencode.exe`, cmdline contains
    `--port <oc_port>`). Killing only the bwrap wrapper orphans the inner opencode
    (it keeps running at 100% CPU). Each episode uses a unique opencode port and
    everything runs as root, so port + (bwrap|opencode) is an unambiguous match."""
    pids = []
    want = f"--port {oc_port}"
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode(errors="ignore")
        except Exception:
            continue
        if want in cmd and ("opencode" in cmd or "bwrap" in cmd):
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("Uid:"):
                            if line.split()[1] == "0":  # root
                                pids.append(pid)
                            break
            except Exception:
                pass
    return pids


def kill_sandbox_proc(proc, oc_port=None):
    """Terminate a sandboxed opencode (root) via sudo. Kills the bwrap wrapper AND
    the opencode inside its namespace (found by port) so nothing is orphaned."""
    # 1) the bwrap wrapper we launched
    if proc is not None:
        try:
            subprocess.run(["sudo", "-n", "kill", "-9", str(proc.pid)], timeout=10,
                           capture_output=True)
        except Exception:
            pass
    # 2) the namespaced opencode + any other root proc bound to this port
    if oc_port:
        for pid in _root_pids_with_port(oc_port):
            try:
                subprocess.run(["sudo", "-n", "kill", "-9", pid], timeout=10, capture_output=True)
            except Exception:
                pass
    if proc is not None:
        try:
            proc.wait(timeout=8)
        except Exception:
            pass
