"""GRPO training helpers using TRL's GRPOTrainer."""
from __future__ import annotations

import json

from datasets import Dataset
from omegaconf import DictConfig
from trl import GRPOConfig

SYSTEM_PROMPT = ""


def build_tool_call_schema(env_name: str = "l2rpn_case14_sandbox") -> dict:
    """Build a JSON schema for tool calls with bounds from the grid environment."""
    import grid2op
    env = grid2op.make(env_name)
    n_line = env.n_line
    n_gen = env.action_space.n_gen
    n_load = env.action_space.n_load
    n_storage = env.action_space.n_storage
    max_element_id = max(n_line, n_gen, n_load) - 1
    env.close()

    actions: list[dict] = [
        {"type": "object", "properties": {"name": {"const": "do_nothing"}, "arguments": {"type": "object", "properties": {}, "additionalProperties": False}}, "required": ["name", "arguments"]},
        {"type": "object", "properties": {"name": {"const": "set_line_status"}, "arguments": {"type": "object", "properties": {"line_id": {"type": "integer", "minimum": 0, "maximum": n_line - 1}, "status": {"type": "string", "enum": ["connect", "disconnect"]}}, "required": ["line_id", "status"], "additionalProperties": False}}, "required": ["name", "arguments"]},
        {"type": "object", "properties": {"name": {"const": "change_bus"}, "arguments": {"type": "object", "properties": {"element_type": {"type": "string", "enum": ["load", "gen", "line_or", "line_ex"]}, "element_id": {"type": "integer", "minimum": 0, "maximum": max_element_id}, "bus": {"type": "integer", "enum": [1, 2]}}, "required": ["element_type", "element_id", "bus"], "additionalProperties": False}}, "required": ["name", "arguments"]},
        {"type": "object", "properties": {"name": {"const": "redispatch"}, "arguments": {"type": "object", "properties": {"gen_id": {"type": "integer", "minimum": 0, "maximum": n_gen - 1}, "delta_mw": {"type": "number"}}, "required": ["gen_id", "delta_mw"], "additionalProperties": False}}, "required": ["name", "arguments"]},
        {"type": "object", "properties": {"name": {"const": "curtail"}, "arguments": {"type": "object", "properties": {"gen_id": {"type": "integer", "minimum": 0, "maximum": n_gen - 1}, "max_mw": {"type": "number"}}, "required": ["gen_id", "max_mw"], "additionalProperties": False}}, "required": ["name", "arguments"]},
    ]
    if n_storage > 0:
        actions.append(
            {"type": "object", "properties": {"name": {"const": "storage"}, "arguments": {"type": "object", "properties": {"storage_id": {"type": "integer", "minimum": 0, "maximum": n_storage - 1}, "mw": {"type": "number"}}, "required": ["storage_id", "mw"], "additionalProperties": False}}, "required": ["name", "arguments"]},
        )
    return {"oneOf": actions}


def build_structural_tag(env_name: str = "l2rpn_case14_sandbox") -> str:
    """Build the structural_tag JSON string for constrained decoding."""
    return json.dumps({
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "\n<tool_call>\n",
            "end": "\n</tool_call>",
            "content": {"type": "json_schema", "json_schema": build_tool_call_schema(env_name)},
        },
    })


_DEFAULT_ENV = "l2rpn_case14_sandbox"
_cached_schema: dict | None = None
_cached_tag: str | None = None


def get_default_tool_call_schema() -> dict:
    """Return the tool call schema for the default environment (lazy-cached)."""
    global _cached_schema
    if _cached_schema is None:
        _cached_schema = build_tool_call_schema(_DEFAULT_ENV)
    return _cached_schema


def get_default_structural_tag() -> str:
    """Return the structural tag for the default environment (lazy-cached)."""
    global _cached_tag
    if _cached_tag is None:
        _cached_tag = build_structural_tag(_DEFAULT_ENV)
    return _cached_tag


def suppress_tool_definitions(trainer) -> None:
    """Patch the trainer's tokenizer to omit tool schemas from prompts.

    TRL auto-injects verbose tool definitions into every prompt via
    apply_chat_template(tools=self.tools). With structural_tag constraining
    output, those definitions are dead weight. This patches the tokenizer
    to always pass tools=None while keeping self.tools populated for
    TRL's tool dispatch loop.
    """
    orig = trainer.processing_class.apply_chat_template

    def _no_tools(*args, tools=None, **kwargs):
        return orig(*args, tools=None, **kwargs)

    trainer.processing_class.apply_chat_template = _no_tools


def build_grpo_config(cfg: DictConfig, output_dir: str | None = None) -> GRPOConfig:
    """Translate Hydra config into a TRL GRPOConfig."""
    t = cfg.training
    env_name = cfg.env.env_name
    structural_tag = build_structural_tag(env_name)
    num_generations = int(t.get("num_generations", 8))
    report_to = str(t.get("report_to", "tensorboard"))

    kwargs = {}
    if report_to == "wandb":
        run_name = str(t.get("run_name", f"gridzero-{env_name}"))
        kwargs["run_name"] = run_name

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
        report_to=report_to,
        use_vllm=bool(t.get("use_vllm", True)),
        vllm_mode=str(t.get("vllm_mode", "colocate")),
        vllm_gpu_memory_utilization=float(t.get("vllm_gpu_memory_utilization", 0.3)),
        vllm_max_model_length=int(t.get("vllm_max_model_length", 1024)),
        chat_template_kwargs={"enable_thinking": False},
        generation_kwargs={"structured_outputs": {"structural_tag": structural_tag}},
        seed=int(cfg.get("seed", 42)),
        bf16=True,
        log_completions=bool(t.get("log_completions", True)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        warmup_steps=int(t.get("warmup_steps", 0)),
        **kwargs,
    )


def get_n_chronics(env_name: str) -> int:
    """Return the number of chronics available in the given grid2op environment."""
    import grid2op
    env = grid2op.make(env_name)
    n = len(env.chronics_handler.real_data.subpaths)
    env.close()
    return n


def build_dataset(cfg: DictConfig) -> Dataset:
    """Build a prompt-only dataset for GRPO training.

    Each row contains the system prompt and a chronics_id for deterministic
    grid2op initialization. The GridEnv receives chronics_id via reset(**kwargs).
    """
    n = int(cfg.training.get("dataset_size", 256))
    n_chronics = get_n_chronics(cfg.env.env_name)
    return Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": ""}]
            for _ in range(n)
        ],
        "chronics_id": [i % n_chronics for i in range(n)],
    })
