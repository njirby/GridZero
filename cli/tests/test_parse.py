import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI = os.path.join(ROOT, "cli", "simctl")


def _load():
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("simctl", CLI)
    spec = importlib.util.spec_from_loader("simctl", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


simctl = _load()


def test_compact_observe_shape():
    s = json.load(open(os.path.join(ROOT, "contracts", "examples", "grid-state-t50.json")))
    out = simctl.compact_observe(s)
    assert "t=50" in out
    assert "lines_down=1" in out
    assert "top loads:" in out
    assert "since your last act" in out
    # overloaded line (1_4_4, rho 1.17) should be flagged
    assert "1_4_4" in out


def test_compact_observe_no_last_action():
    s = json.load(open(os.path.join(ROOT, "contracts", "examples", "grid-state-t0.json")))
    out = simctl.compact_observe(s)
    assert "since your last act" not in out
    assert "done=no" in out


def test_compact_observe_empty_lines():
    out = simctl.compact_observe({"t": 0, "reward": -10, "cum_reward": 0, "done": False,
                                  "n_down": 0, "lines": [], "gens": [], "last_action": None})
    assert "t=0" in out
    assert "-" in out  # maxline fallback
