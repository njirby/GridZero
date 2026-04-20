"""Tokenizer utilities and prompt templates for tool call generation."""
from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizer

# The system message frames the task; actual grid state arrives via encoder embeddings
# injected by vllm as multimodal prefix tokens before the assistant turn.
SYSTEM_PROMPT = (
    "You are an autonomous power grid operator. "
    "You will receive the current grid state as context and must output a single "
    "JSON tool call to control the grid. Output only valid JSON, nothing else."
)

# Qwen3 chat template — grid state embeddings are injected by vllm in place of
# the [GRID STATE] placeholder tokens.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n[GRID STATE]<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def get_tokenizer(model_name: str = "Qwen/Qwen3-4B") -> PreTrainedTokenizer:
    """Load the tokenizer for the policy model."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def build_prompt(system: str = SYSTEM_PROMPT) -> str:
    """Return the formatted prompt string.

    Grid state embeddings are injected by vllm separately — [GRID STATE]
    is a placeholder that the vllm multimodal embedding injection replaces.
    """
    return PROMPT_TEMPLATE.format(system=system)


def get_eos_token_id(tokenizer: PreTrainedTokenizer) -> int:
    """Return the EOS token ID used to terminate generation."""
    return tokenizer.eos_token_id
