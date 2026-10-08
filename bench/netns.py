"""bench/netns.py — per-episode network isolation for benchmark runs.

WHY: the sandbox shares the host network namespace, so an (adversarial) agent
could: read/tamper sibling episode backends over localhost, reach the
interactive backend (no token), call vLLM directly (off-trace token burn /
oracle use), and hit the internet via raw IP (DNS is dead, TCP is not).

How: each episode gets its own netns (a `sleep` holder keeps it alive; the
backend and the sandboxed opencode both run inside it via `nsenter`). A veth
pair connects it to the host:
  - host side  : 10.200.<sub>.1  (gateway for the netns; sub = port % 250)
  - netns side : 10.200.<sub>.2  (the backend listens on 0.0.0.0:port here)
The ONLY external service the agent needs is vLLM: a tiny host-side TCP
forwarder on the gateway IP proxies <gw>:8002 -> 127.0.0.1:8002 (no NAT).
Host INPUT admits ONLY tcp <gw>:<vllm_port> from the episode veth and
DROPs everything else arriving on it, so every other host service on the
gateway IP (8001 raw vLLM, 8731, sibling backends, ...) and the internet are
unreachable.

Inside the netns, localhost still works for the two in-netns services
(backend at 127.0.0.1:port, opencode at 127.0.0.1:oc_port), so simctl /
driver need no URL changes.
"""
from __future__ import annotations
import os, re, subprocess, time

PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv/bin/python")

_FORWARDER = r"""
import socket, sys, threading
gw, lport, rport = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
l = socket.socket()
l.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
l.bind((gw, lport)); l.listen(128)
def pump(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d: break
            b.sendall(d)
    except Exception: pass
    for s in (b, a):
        try: s.shutdown(socket.SHUT_WR)
        except Exception: pass
while True:
    c, _ = l.accept()
    try:
        r = socket.create_connection(("127.0.0.1", rport), timeout=5)
    except Exception:
        c.close(); continue
    threading.Thread(target=pump, args=(c, r), daemon=True).start()
    threading.Thread(target=pump, args=(r, c), daemon=True).start()
"""


class EpisodeNet:
    def __init__(self, port, vllm_port=8002, log_dir=None):
        self.port = port
        self.vllm_port = vllm_port
        self.sub = port % 250  # NOTE: two CONCURRENT episodes whose ports share
        # port%250 would collide on the subnet; bench runners use unique ports.
        self.gw = f"10.200.{self.sub}.1"
        self.inside = f"10.200.{self.sub}.2"
        self.h_veth = f"gzth{port}"
        self.s_veth = f"gzts{port}"
        self.name = f"gzns{port}"  # named netns (ip netns) — no holder PID to track
        self.forwarder = None
        self._log = None
        self.log_dir = log_dir

    # ---- netns helpers ----
    def _run(self, cmd):
        return subprocess.run(["sudo", "-n", "-E", "ip", "netns", "exec", self.name] + cmd,
                              check=True, capture_output=True, text=True)

    def ns(self, argv):
        """Prefix argv so it runs (as root) inside this episode's netns."""
        return ["sudo", "-n", "-E", "ip", "netns", "exec", self.name] + argv

    @staticmethod
    def _kill_listener(addr_port):
        """Kill any process bound to <addr>:<port> (a stale forwarder orphaned
        from a SIGKILL'd runner would otherwise wedge the next setup)."""
        try:
            out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
        except Exception:
            return
        for line in out.splitlines():
            if f":{addr_port.rsplit(':', 1)[1]} " in line and addr_port.rsplit(':', 1)[0] in line:
                m = re.search(r"pid=(\d+)", line)
                if m:
                    subprocess.run(["sudo", "-n", "kill", "-9", m.group(1)], capture_output=True)

    def _fw_rules(self):
        """(accept, drop) iptables rule specs this episode adds to INPUT."""
        accept = ["-i", self.h_veth, "-s", self.inside, "-d", self.gw, "-p", "tcp",
                  "--dport", str(self.vllm_port), "-j", "ACCEPT"]
        drop = ["-i", self.h_veth, "-j", "DROP"]
        return accept, drop

    def _del_fw_rules(self):
        """Delete exactly the rules setup() adds (every copy, so stale ones go too)."""
        for rule in self._fw_rules():
            while subprocess.run(["sudo", "-n", "iptables", "-D", "INPUT"] + rule,
                                 capture_output=True).returncode == 0:
                pass

    # ---- lifecycle ----
    def setup(self):
        # best-effort cleanup of a stale netns/pair from a crashed run
        subprocess.run(["sudo", "-n", "ip", "netns", "del", self.name], capture_output=True)
        for name in (self.h_veth, self.s_veth):
            subprocess.run(["sudo", "-n", "ip", "link", "del", name], capture_output=True)
        subprocess.run(["sudo", "-n", "ip", "netns", "add", self.name], check=True,
                       capture_output=True)
        subprocess.run(["sudo", "-n", "ip", "link", "add", self.h_veth, "type", "veth",
                        "peer", "name", self.s_veth], check=True, capture_output=True)
        subprocess.run(["sudo", "-n", "ip", "addr", "add", f"{self.gw}/24", "dev", self.h_veth],
                       check=True, capture_output=True)
        subprocess.run(["sudo", "-n", "ip", "link", "set", self.h_veth, "up"],
                       check=True, capture_output=True)
        subprocess.run(["sudo", "-n", "ip", "link", "set", self.s_veth, "netns", self.name],
                       check=True, capture_output=True)
        self._run(["ip", "link", "set", "lo", "up"])
        self._run(["ip", "addr", "add", f"{self.inside}/24", "dev", self.s_veth])
        self._run(["ip", "link", "set", self.s_veth, "up"])
        self._run(["ip", "route", "add", "default", "via", self.gw])
        # host INPUT: admit ONLY the vLLM forwarder port on the gw IP from this
        # veth, then DROP everything else arriving on it (the sandbox shares the
        # gateway IP with every host service listening on 0.0.0.0).
        self._del_fw_rules()  # stale rules from a crashed run
        accept, drop = self._fw_rules()
        subprocess.run(["sudo", "-n", "iptables", "-I", "INPUT", "1"] + drop,
                       check=True, capture_output=True)
        subprocess.run(["sudo", "-n", "iptables", "-I", "INPUT", "1"] + accept,
                       check=True, capture_output=True)
        # the only external service the agent gets: vLLM via the gateway IP
        self._kill_listener(f"{self.gw}:{self.vllm_port}")
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._log = open(os.path.join(self.log_dir, f"netns-forward-{self.port}.log"), "ab")
        self.forwarder = subprocess.Popen([PY, "-c", _FORWARDER, self.gw,
                                           str(self.vllm_port), str(self.vllm_port)],
                                          stdout=self._log, stderr=subprocess.STDOUT)
        time.sleep(0.5)
        if self.forwarder.poll() is not None:
            raise RuntimeError("vLLM forwarder died during setup")

    def backend_base(self):
        return f"http://{self.inside}:{self.port}"

    def vllm_url(self):
        return f"http://{self.gw}:{self.vllm_port}"

    def teardown(self):
        if self.forwarder:
            try:
                self.forwarder.terminate()
                self.forwarder.wait(timeout=5)
            except Exception:
                try:
                    self.forwarder.kill()
                except Exception:
                    pass
        # del the netns first (drops the veth peer inside it), then the host veth
        subprocess.run(["sudo", "-n", "ip", "netns", "del", self.name], capture_output=True)
        subprocess.run(["sudo", "-n", "ip", "link", "del", self.h_veth], capture_output=True)
        self._del_fw_rules()
        self._kill_listener(f"{self.gw}:{self.vllm_port}")  # orphaned-forwarder sweep
        if self._log:
            try:
                self._log.close()
            except Exception:
                pass
