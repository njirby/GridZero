"""Tests for TRL config and dataset construction."""
from __future__ import annotations

import json

import grid2op
from omegaconf import OmegaConf

from gridzero.training.gspo import (
    build_dataset,
    build_grpo_config,
    build_tool_call_schema,
    get_default_structural_tag,
    get_default_tool_call_schema,
)


def _cfg():
    return OmegaConf.create({
        "seed": 42,
        "output_dir": "outputs/test",
        "env": {"env_name": "l2rpn_case14_sandbox"},
        "policy": {"model_name": "Qwen/Qwen3-0.6B"},
        "training": {
            "num_generations": 4,
            "max_completion_length": 256,
            "max_tool_calling_iterations": 1,
            "loss_type": "grpo",
            "epsilon": 0.2,
            "beta": 0.04,
            "learning_rate": 1e-5,
            "max_grad_norm": 1.0,
            "max_steps": 10,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "logging_steps": 1,
            "save_steps": 5,
            "report_to": "none",
            "use_vllm": False,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": 0.3,
            "dataset_size": 16,
            "n_chronics": 100,
        },
    })


def test_build_grpo_config():
    cfg = _cfg()
    grpo = build_grpo_config(cfg)
    assert grpo.num_generations == 4
    assert grpo.max_completion_length == 256
    assert grpo.loss_type == "grpo"
    assert grpo.epsilon == 0.2
    assert grpo.beta == 0.04
    assert grpo.learning_rate == 1e-5
    assert grpo.max_steps == 10
    assert grpo.seed == 42


def test_build_grpo_config_output_dir_override():
    cfg = _cfg()
    grpo = build_grpo_config(cfg, output_dir="/tmp/custom")
    assert grpo.output_dir == "/tmp/custom"


def test_build_dataset_schema():
    cfg = _cfg()
    ds = build_dataset(cfg)
    assert "prompt" in ds.column_names
    assert "chronics_id" in ds.column_names
    assert len(ds) == 16


def test_dataset_prompt_is_message_list():
    cfg = _cfg()
    ds = build_dataset(cfg)
    prompt = ds[0]["prompt"]
    assert isinstance(prompt, list)
    assert len(prompt) == 1
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"] == ""


def test_dataset_chronics_id_wraps():
    cfg = _cfg()
    cfg.training.dataset_size = 200
    cfg.training.n_chronics = 50
    ds = build_dataset(cfg)
    assert ds[50]["chronics_id"] == 0
    assert ds[99]["chronics_id"] == 49


def test_grpo_config_has_constrained_generation():
    cfg = _cfg()
    grpo = build_grpo_config(cfg)
    so = grpo.generation_kwargs["structured_outputs"]
    assert "structural_tag" in so


def test_structural_tag_compiles_with_xgrammar():
    import xgrammar as xgr
    grammar = xgr.Grammar.from_structural_tag(get_default_structural_tag())
    assert grammar is not None


def test_tool_call_schema_covers_all_actions():
    schema = get_default_tool_call_schema()
    names = {s["properties"]["name"]["const"] for s in schema["oneOf"]}
    expected = {"do_nothing", "set_line_status", "change_bus", "redispatch", "curtail"}
    assert names == expected


def test_tool_call_schema_has_bounds():
    schema = get_default_tool_call_schema()
    for action in schema["oneOf"]:
        props = action["properties"]["arguments"]["properties"]
        for key, spec in props.items():
            if spec.get("type") == "integer" and key not in ("bus",):
                assert "minimum" in spec, f"{key} missing minimum"
                assert "maximum" in spec, f"{key} missing maximum"


def test_schema_matches_env_dimensions():
    env = grid2op.make("l2rpn_case14_sandbox")
    schema = build_tool_call_schema("l2rpn_case14_sandbox")
    actions = {s["properties"]["name"]["const"]: s for s in schema["oneOf"]}

    line_max = actions["set_line_status"]["properties"]["arguments"]["properties"]["line_id"]["maximum"]
    assert line_max == env.n_line - 1

    gen_max = actions["redispatch"]["properties"]["arguments"]["properties"]["gen_id"]["maximum"]
    assert gen_max == env.action_space.n_gen - 1

    assert "storage" not in actions
    env.close()
