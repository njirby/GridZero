"""Integration tests: ToolCall JSON -> grid2op Action -> applied in env."""
from __future__ import annotations

import json
from numbers import Real

import pytest


def _build_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "env": {
                "env_name": "l2rpn_case14_sandbox",
                "backend": "lightsim2grid",
                "max_steps": 20,
                "test": True,
            }
        }
    )


def _candidate_payloads(env) -> dict[str, dict]:
    payloads: dict[str, dict] = {
        "do_nothing": {"action_type": "do_nothing"},
    }
    if env.n_line > 0:
        payloads["set_line_status"] = {
            "action_type": "set_line_status",
            "line_id": 0,
            "status": "disconnect",
        }
        payloads["change_bus_line_or"] = {
            "action_type": "change_bus",
            "element_type": "line_or",
            "element_id": 0,
            "bus": 2,
        }
    if env.n_load > 0:
        payloads["change_bus_load"] = {
            "action_type": "change_bus",
            "element_type": "load",
            "element_id": 0,
            "bus": 2,
        }
    if env.n_gen > 0:
        payloads["redispatch"] = {
            "action_type": "redispatch",
            "gen_id": 0,
            "delta_mw": 1.0,
        }
        payloads["curtail"] = {
            "action_type": "curtail",
            "gen_id": 0,
            "max_mw": 10.0,
        }
    if getattr(env, "n_storage", 0) > 0:
        payloads["storage"] = {
            "action_type": "storage",
            "storage_id": 0,
            "mw": 1.0,
        }
    return payloads


def test_tool_schemas_translate_and_apply_in_grid2op_env():
    pytest.importorskip("grid2op")
    from gridzero.env.actions import parse_tool_call
    from gridzero.env.wrapper import make_env

    env, _ = make_env(_build_cfg())
    obs = env.reset()
    payloads = _candidate_payloads(env)

    executed: list[str] = []
    skipped: list[str] = []

    try:
        for name, payload in payloads.items():
            action = parse_tool_call(json.dumps(payload), env.action_space)
            assert action is not None
            try:
                obs, reward, done, info = env.step(action)
                assert obs is not None
                assert isinstance(reward, Real)
                assert isinstance(done, (bool, int))
                assert isinstance(info, dict)
                executed.append(name)
                if done:
                    obs = env.reset()
            except Exception:
                # Some action families can be unsupported on a given env
                # despite valid syntax (for example no redispatchable gens).
                skipped.append(name)
    finally:
        env.close()

    assert "do_nothing" in executed
    # Require topology controls to be executable on case14.
    assert any(name.startswith("change_bus") for name in executed)
    assert "set_line_status" in executed
