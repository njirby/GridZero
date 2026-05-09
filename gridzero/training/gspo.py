"""GRPO training helpers using TRL's GRPOTrainer."""
from __future__ import annotations

import json

from datasets import Dataset
from omegaconf import DictConfig
from trl import GRPOConfig

SYSTEM_PROMPT = (
    "You are an autonomous power grid operator for a 14-substation network with "
    "20 powerlines, 6 generators, and 11 loads. Given the current grid state, call "
    "exactly one tool to control the grid. Your goal is to keep all lines below "
    "thermal limits and prevent blackouts."
)

TOOL_CALL_SCHEMA: dict = {"oneOf": [
    {"type": "object", "properties": {"name": {"const": "do_nothing"}, "arguments": {"type": "object", "properties": {}, "additionalProperties": False}}, "required": ["name", "arguments"]},
    {"type": "object", "properties": {"name": {"const": "set_line_status"}, "arguments": {"type": "object", "properties": {"line_id": {"type": "integer"}, "status": {"type": "string", "enum": ["connect", "disconnect"]}}, "required": ["line_id", "status"], "additionalProperties": False}}, "required": ["name", "arguments"]},
    {"type": "object", "properties": {"name": {"const": "change_bus"}, "arguments": {"type": "object", "properties": {"element_type": {"type": "string", "enum": ["load", "gen", "line_or", "line_ex"]}, "element_id": {"type": "integer"}, "bus": {"type": "integer", "enum": [1, 2]}}, "required": ["element_type", "element_id", "bus"], "additionalProperties": False}}, "required": ["name", "arguments"]},
    {"type": "object", "properties": {"name": {"const": "redispatch"}, "arguments": {"type": "object", "properties": {"gen_id": {"type": "integer"}, "delta_mw": {"type": "number"}}, "required": ["gen_id", "delta_mw"], "additionalProperties": False}}, "required": ["name", "arguments"]},
    {"type": "object", "properties": {"name": {"const": "curtail"}, "arguments": {"type": "object", "properties": {"gen_id": {"type": "integer"}, "max_mw": {"type": "number"}}, "required": ["gen_id", "max_mw"], "additionalProperties": False}}, "required": ["name", "arguments"]},
    {"type": "object", "properties": {"name": {"const": "storage"}, "arguments": {"type": "object", "properties": {"storage_id": {"type": "integer"}, "mw": {"type": "number"}}, "required": ["storage_id", "mw"], "additionalProperties": False}}, "required": ["name", "arguments"]},
]}

STRUCTURAL_TAG: str = json.dumps({
    "type": "structural_tag",
    "format": {
        "type": "tag",
        "begin": "\n<tool_call>\n",
        "end": "\n</tool_call>",
        "content": {"type": "json_schema", "json_schema": TOOL_CALL_SCHEMA},
    },
})


def build_grpo_config(cfg: DictConfig, output_dir: str | None = None) -> GRPOConfig:
    """Translate Hydra config into a TRL GRPOConfig."""
    t = cfg.training
    num_generations = int(t.get("num_generations", 8))
    return GRPOConfig(
        output_dir=output_dir or cfg.output_dir,
        num_generations=num_generations,
        generation_batch_size=num_generations,
        max_completion_length=int(t.get("max_completion_length", 512)),
        max_tool_calling_iterations=int(t.get("max_tool_calling_iterations", 1)),
        loss_type=str(t.get("loss_type", "grpo")),
        epsilon=float(t.get("epsilon", 0.2)),
        beta=float(t.get("beta", 0.04)),
        learning_rate=float(t.get("learning_rate", 1e-5)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        max_steps=int(t.get("max_steps", 1000)),
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 4)),
        logging_steps=int(t.get("logging_steps", 1)),
        save_steps=int(t.get("save_steps", 250)),
        report_to=str(t.get("report_to", "tensorboard")),
        use_vllm=bool(t.get("use_vllm", True)),
        vllm_mode=str(t.get("vllm_mode", "colocate")),
        vllm_gpu_memory_utilization=float(t.get("vllm_gpu_memory_utilization", 0.3)),
        chat_template_kwargs={"enable_thinking": False},
        generation_kwargs={"structured_outputs": {"structural_tag": STRUCTURAL_TAG}},
        seed=int(cfg.get("seed", 42)),
        bf16=True,
        log_completions=bool(t.get("log_completions", True)),
    )


def build_dataset(cfg: DictConfig) -> Dataset:
    """Build a prompt-only dataset for GRPO training.

    Each row contains the system prompt and a chronics_id for deterministic
    grid2op initialization. The GridEnv receives chronics_id via reset(**kwargs).
    """
    n = int(cfg.training.get("dataset_size", 256))
    n_chronics = int(cfg.training.get("n_chronics", 1004))
    return Dataset.from_dict({
        "prompt": [
            [{"role": "system", "content": SYSTEM_PROMPT}]
            for _ in range(n)
        ],
        "chronics_id": [i % n_chronics for i in range(n)],
    })
