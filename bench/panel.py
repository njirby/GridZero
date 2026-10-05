"""bench/panel.py — load the fixed evaluation panel + hash the benchmark config.

Reproducibility (per R2: model-version drift has inverted study conclusions) —
every run record carries a config hash of the panel + env + sampling + harness
version, so results are comparable across time and auditable.
"""
import hashlib, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(ROOT, "bench", "panel.json")


def load_panel(path=PANEL_PATH):
    return json.load(open(path))


def config_hash(model="qwen3.5-4b", horizon=None, panel=None, agent="build",
                temperature=0.0, harness_version="0.1"):
    panel = panel or load_panel()
    h = hashlib.sha256()
    blob = {
        "model": model, "agent": agent, "temperature": temperature,
        "env": panel["env"], "horizons": panel["horizons"],
        "horizon": horizon if horizon is not None else panel["horizons"]["pilot"],
        "chronics_pilot": panel["pilot_chronics"],
        "chronics_standard": panel["standard_chronics"],
        "sampling": panel["sampling"],
        "harness_version": harness_version,
    }
    h.update(json.dumps(blob, sort_keys=True).encode())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    p = load_panel()
    print("env:", p["env"])
    print("horizons:", p["horizons"])
    print("pilot chronics:", p["pilot_chronics"])
    print("standard chronics (n=%d):" % len(p["standard_chronics"]), p["standard_chronics"])
    print("config hash (pilot, AA-Dense-Blackwell):", config_hash(horizon=p["horizons"]["pilot"], panel=p))
    print("config hash (standard, AA-Dense-Blackwell):", config_hash(horizon=p["horizons"]["standard"], panel=p))
