"""Offline unit tests for the bench harness fixes (no backend, GPU, sudo or docker)."""
import json, os, sys
from types import SimpleNamespace
from unittest import mock
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bench import run_2agent, run_llm, report
from bench.netns import EpisodeNet
from bench.panel import config_hash
from bench.score import EpisodeResult


def _er(agent="DoNothing", chronic=0, horizon=96, cum=100.0, survived=96, done=True, **kw):
    return EpisodeResult(agent=agent, chronic=chronic, horizon=horizon, survived=survived,
                         done=done, cum_reward=cum, **kw)


# ---- #1 run_2agent unpacks start_backend and tears everything down ----
def _run_2agent_main(tmp_path, monkeypatch):
    proc, net = mock.Mock(), mock.Mock()
    net.backend_base.return_value = "http://10.200.7.2:8950"
    monkeypatch.setattr(run_2agent, "ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run_2agent.py", "--port", "8950"])
    seen = {}

    def fake_wait_up(base, path, timeout=0):
        seen["base"] = base
        return False  # backend "never boots" -> early return, teardown must still run
    sb = mock.Mock()
    with mock.patch.object(run_2agent, "start_backend", return_value=(proc, "tok", net)) as sbk, \
            mock.patch.object(run_2agent, "wait_up", fake_wait_up), \
            mock.patch("bench.sandbox.kill_sandbox_proc", sb):
        rc = run_2agent.main()
    return rc, proc, net, seen, sbk, sb


def test_2agent_uses_netns_base_and_tears_down(tmp_path, monkeypatch):
    rc, proc, net, seen, sbk, sb = _run_2agent_main(tmp_path, monkeypatch)
    assert rc == 2
    assert seen["base"] == "http://10.200.7.2:8950"
    proc.terminate.assert_called_once()
    net.teardown.assert_called_once()
    assert sb.call_count == 2


def test_2agent_teardown_failures_are_visible(capsys):
    proc, net = mock.Mock(), mock.Mock()
    proc.terminate.side_effect = RuntimeError("boom")
    net.teardown.side_effect = RuntimeError("nsboom")
    with mock.patch("bench.sandbox.kill_sandbox_proc"):
        run_2agent.teardown_backend(proc, net, 8950)
    out = capsys.readouterr().out
    assert "boom" in out and "nsboom" in out
    net.teardown.assert_called_once()  # a failing terminate must not skip the netns


# ---- #3 netns failure fails the episode unless opted out ----
def _start(tmp_path, monkeypatch, allow):
    (tmp_path / "runs").mkdir(exist_ok=True)
    monkeypatch.setattr(run_llm, "ROOT", str(tmp_path))
    monkeypatch.setenv("OPENCODE_NETNS", "1")
    with mock.patch("bench.netns.EpisodeNet.setup", side_effect=RuntimeError("no sudo")), \
            mock.patch("bench.netns.EpisodeNet.teardown"), \
            mock.patch.object(run_llm.subprocess, "Popen") as popen:
        return run_llm.start_backend(8801, {}, allow_no_netns=allow), popen


def test_netns_failure_is_fatal_by_default(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="allow-no-netns"):
        _start(tmp_path, monkeypatch, allow=False)


def test_netns_failure_opt_out_marks_unisolated(tmp_path, monkeypatch):
    (_, tok, net), popen = _start(tmp_path, monkeypatch, allow=True)
    assert net is None and tok
    popen.assert_called_once()


def test_build_result_records_isolated():
    st = {"sim": {"t": 3}, "llm": {}}
    for iso in (True, False):
        r = run_llm.build_result(SimpleNamespace(model="m", chronic=1, isolated=iso), 0, 10, st, 1.0, "x")
        assert r.isolated is iso


# ---- #2 firewall rules ----
def test_fw_rules_restrict_to_forwarder_port():
    accept, drop = EpisodeNet(8801, vllm_port=8002)._fw_rules()
    assert "ACCEPT" in accept and accept[accept.index("--dport") + 1] == "8002"
    assert "-d" in accept and drop[-1] == "DROP" and "-s" not in drop


# ---- #5 report accepts several files per flag / repeated flags ----
def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")


@pytest.mark.parametrize("style", ["nargs", "repeat"])
def test_report_multiple_inputs(tmp_path, monkeypatch, style):
    b1, b2, l1, l2 = (str(tmp_path / n) for n in ("b1.jsonl", "b2.jsonl", "l1.jsonl", "l2.jsonl"))
    _write(b1, [_er()])
    _write(b2, [_er(chronic=1)])
    _write(l1, [_er("llm", cum=110.0)])
    _write(l2, [_er("llm", chronic=1, cum=120.0)])
    argv = (["--baselines", b1, b2, "--llm", l1, l2] if style == "nargs"
            else ["--baselines", b1, "--baselines", b2, "--llm", l1, "--llm", l2])
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["report.py"] + argv + ["--out", out])
    report.main()
    summ = json.load(open(os.path.join(out, "summary.json")))
    assert summ["agents"]["llm"]["n_episodes"] == 2 and summ["n_anchors"] == 2


# ---- #6 anchors: mean over repeats; unanchored rows are reported ----
def test_build_anchors_is_mean():
    a = report.build_anchors([_er(cum=100.0), _er(cum=200.0), _er(chronic=1, cum=5.0)])
    assert a[(0, 96)].cum_reward == 150.0 and a[(1, 96)].cum_reward == 5.0


def test_report_warns_on_dropped_rows(tmp_path, monkeypatch, capsys):
    f = str(tmp_path / "r.jsonl")
    _write(f, [_er(), _er("llm", chronic=7, ep="ep-orphan")])
    monkeypatch.setattr(sys, "argv", ["report.py", "--baselines", f, "--out", str(tmp_path / "o")])
    report.main()
    err = capsys.readouterr().err
    assert "DROPPED" in err and "ep-orphan" in err


# ---- #7 config hash covers ablation / seed / adversarial ----
def test_config_hash_sensitivity():
    base = config_hash(horizon=96)
    assert config_hash(horizon=96, ablation={"docs": False}) != base
    assert config_hash(horizon=96, ablation={"doc_warning": "x"}) != config_hash(
        horizon=96, ablation={"doc_warning": "y"})
    assert config_hash(horizon=96, seed=1) != config_hash(horizon=96, seed=2)
    adv = {"interval": 96, "duration": 24, "seed": 0}
    assert config_hash(horizon=96, adversarial=adv) != base
    assert config_hash(horizon=96, adversarial=dict(adv, interval=48)) != config_hash(horizon=96, adversarial=adv)
    assert config_hash(horizon=96, seed=1) == config_hash(horizon=96, seed=1)


# ---- #8 chronic id must not reach the model ----
def test_kickoff_hides_chronic():
    k = run_llm.kickoff_prompt(742, 96, {})
    assert "742" not in k and "chronic" not in k.lower()
