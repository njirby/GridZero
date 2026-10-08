"""Do-nothing (DN) baseline per (chronic, horizon, seed), computed with the backend's own
SimSession and the exact reset the backend does at startup from SIM_CHRONIC/SIM_HORIZON/
SIM_SEED (backend/app/main.py). The RL verifier scores cum_reward - dn_cum_reward.

Usage: .venv/bin/python lego/baselines/compute_dn.py --chronic 0-319,996-1003 --horizon 24 --seed 0
"""
import argparse, json, os, sys, tempfile, warnings
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def parse_ranges(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


_SESS = None


def run_one(args):
    global _SESS
    warnings.filterwarnings("ignore")
    from backend.app.sim_session import SimSession
    chronic, horizon, seed = args
    if _SESS is None:
        _SESS = SimSession(render_dir=tempfile.mkdtemp())
    s = _SESS
    s.reset(seed=seed, options={"time serie id": chronic, "max step": horizon})
    out = None
    while not s._finished:
        out, _ = s.step(1)
    st = s.latest_state()
    return {"chronic": chronic, "horizon": horizon, "seed": seed,
            "dn_cum_reward": round(float(st["cum_reward"]), 4), "dn_t": int(st["t"]),
            "dn_cause": st.get("cause")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chronic", required=True)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [(c, a.horizon, a.seed) for c in parse_ranges(a.chronic)]
    with Pool(a.procs) as p:
        rows = p.map(run_one, jobs, chunksize=4)
    out = a.out or os.path.join(os.path.dirname(__file__), f"dn_h{a.horizon}_s{a.seed}.json")
    json.dump({str(r["chronic"]): r for r in rows}, open(out, "w"), indent=1)
    print(f"wrote {len(rows)} -> {out}")
