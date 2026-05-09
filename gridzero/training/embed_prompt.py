"""Prompt embedding construction for observation-injection training."""
from __future__ import annotations

import torch
import torch.nn as nn


def cache_template_ids(
    tokenizer, system_prompt: str
) -> tuple[list[int], list[int]]:
    """Pre-tokenize the fixed chat template segments around the observation slot.

    Returns (prefix_ids, suffix_ids) where obs embeddings are inserted between them.
    """
    placeholder = "OBS_PLACEHOLDER_XYZZY"
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": placeholder},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prefix_text, suffix_text = text.split(placeholder)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    return prefix_ids, suffix_ids


def _get_embed_fn(model: nn.Module) -> nn.Embedding:
    """Extract the token embedding layer from a HuggingFace model."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    raise AttributeError("Cannot find token embedding layer on model")


def build_generation_embeds(
    prefix_ids: list[int],
    suffix_ids: list[int],
    obs_embeds: torch.Tensor,
    embed_fn: nn.Module,
) -> torch.Tensor:
    """Build the full prompt embedding for vLLM generation.

    Args:
        prefix_ids: Token IDs for system prompt + user role prefix.
        suffix_ids: Token IDs for user end + assistant prefix.
        obs_embeds: Observation embeddings from the encoder, shape [N, hidden_size].
        embed_fn: The model's token embedding layer.

    Returns:
        2D tensor [S+N+T, hidden_size] on CPU, suitable for EmbedsPrompt.
    """
    embed_device = next(embed_fn.parameters()).device
    dtype = obs_embeds.dtype

    prefix_t = torch.tensor(prefix_ids, device=embed_device)
    suffix_t = torch.tensor(suffix_ids, device=embed_device)

    with torch.no_grad():
        prefix_embeds = embed_fn(prefix_t).to(dtype)
        suffix_embeds = embed_fn(suffix_t).to(dtype)

    obs_embeds = obs_embeds.to(embed_device)
    return torch.cat([prefix_embeds, obs_embeds, suffix_embeds], dim=0).cpu()


def build_training_embeds(
    prefix_ids: list[int],
    suffix_ids: list[int],
    obs_embeds: torch.Tensor,
    completion_ids: torch.Tensor,
    embed_fn: nn.Module,
) -> tuple[torch.Tensor, int]:
    """Build full inputs_embeds for the training forward pass.

    The observation embeddings carry gradients so the encoder receives
    the GRPO training signal.

    Args:
        prefix_ids: Token IDs for system prompt + user role prefix.
        suffix_ids: Token IDs for user end + assistant prefix.
        obs_embeds: Observation embeddings [N, hidden_size], with grad.
        completion_ids: Action token IDs [C].
        embed_fn: The model's token embedding layer.

    Returns:
        (inputs_embeds [P+C, hidden_size], prompt_length P)
        where P = len(prefix) + N + len(suffix).
    """
    embed_device = next(embed_fn.parameters()).device
    dtype = obs_embeds.dtype

    prefix_t = torch.tensor(prefix_ids, device=embed_device)
    suffix_t = torch.tensor(suffix_ids, device=embed_device)
    comp_ids = completion_ids.to(embed_device)

    prefix_embeds = embed_fn(prefix_t).to(dtype)
    suffix_embeds = embed_fn(suffix_t).to(dtype)
    completion_embeds = embed_fn(comp_ids).to(dtype)

    obs_embeds = obs_embeds.to(embed_device)
    prompt_length = len(prefix_ids) + obs_embeds.shape[0] + len(suffix_ids)
    inputs_embeds = torch.cat(
        [prefix_embeds, obs_embeds, suffix_embeds, completion_embeds], dim=0
    )
    return inputs_embeds, prompt_length
