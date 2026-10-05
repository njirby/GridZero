#!/usr/bin/env python3
import subprocess, json, time

LOG = open("/tmp/opencode/controller.log", "a", buffering=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, file=LOG)

def simctl(*args):
    r = subprocess.run(["simctl", *args], capture_output=True, text=True, timeout=30)
    try: return json.loads(r.stdout)
    except Exception: return {"ok": False, "error": (r.stdout + r.stderr).strip()}

def observe():
    d = simctl("observe", "--json")
    return d.get("data") if d.get("ok") else None

def act(a):
    d = simctl("act", json.dumps(a))
    log("ACT", json.dumps(a), "->", d.get("_human", d.get("error")))
    return d.get("ok")

EXPORT_LINES = {"5_10_7", "5_11_8", "5_12_9"}   # sub_5 exports -> wind-sensitive
WIND_MIN, WIND_MAX = 0.15, 1.0
EXP_HI, EXP_LO = 0.80, 0.42

log("=== controller v3 start ===")
wind = 0.55
applied = False        # structural 1_4_4 reroute this episode
last_t = -1
iters = 0

while True:
    d = observe()
    if d is None:
        time.sleep(0.5); continue
    t = d.get("t", 0)

    # detect episode reset -> re-apply structural fix, reset state
    if t < last_t - 1:
        log(f"RESET detected {last_t}->{t}; re-init")
        applied = False
    last_t = t

    if d.get("done"):
        iters += 1
        if iters % 20 == 0: log("t=%d DONE cause=%s (waiting for backend reset)" % (t, d.get("cause")))
        time.sleep(1); continue

    lines = {l["name"]: l for l in d["lines"]}

    def rho(n):
        l = lines.get(n)
        return l["rho"] if (l and l["status"] == "up") else None

    acted = False

    # 1) STRUCTURAL: spread sub_4 intake off the tiny 1_4_4 onto big 0_4_1 (validated safe)
    if not applied:
        r144, r041, r133 = rho("1_4_4"), rho("0_4_1"), rho("1_3_3")
        if r144 is not None and r144 > 0.45 and (r041 or 0) < 0.55 and (r133 or 0) < 0.70:
            if act({"change_bus": {"lines_ex_id": ["1_4_4"]}}):
                applied = True

    # 2) EXPORT wind balance (only wind-sensitive lever; no sub_5 change_bus)
    if not acted:
        exp = max([rho(n) for n in EXPORT_LINES if rho(n) is not None], default=0.0)
        if exp >= EXP_HI:
            wind = max(WIND_MIN, round(wind - 0.08, 2))
        elif exp <= EXP_LO:
            wind = min(WIND_MAX, round(wind + 0.08, 2))
        else:
            wind = round(wind, 2)  # hold
        if abs(wind - getattr(controller, "_w", wind)) >= 0.06:
            act({"curtail": {"gen_5_2": wind}}); controller._w = wind
            acted = True

    # 3) SUB_13: restore dual feeder if 12_13_14 down and 8_13_11 hot
    if not acted:
        l81311, l121314 = lines.get("8_13_11"), lines.get("12_13_14")
        if l81311 and l81311["status"] == "up" and l81311["rho"] >= 0.85 and \
           l121314 and l121314["status"] != "up" and l121314["cooldown"] <= 0:
            if act({"set_line_status": {"12_13_14": 1}}):
                acted = True

    iters += 1
    if iters % 12 == 0 or acted:
        hot = [(l["name"], round(l["rho"], 2)) for l in sorted(d["lines"], key=lambda x: -x["rho"])[:3] if l["status"] == "up"]
        log("t=%d max=%.3f wind=%.2f applied=%s down=%d ovf=%d top=%s" %
            (t, d.get("max_rho", 0), wind, applied, d.get("n_down", 0), d.get("n_overflow", 0), hot))

    time.sleep(0.15)

class controller: pass
controller._w = wind
