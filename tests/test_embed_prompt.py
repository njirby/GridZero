"""Tests for prompt embedding construction."""
from __future__ import annotations

import torch
import torch.nn as nn

from gridzero.training.embed_prompt import (
    build_generation_embeds,
    build_training_embeds,
    cache_template_ids,
)
from gridzero.training.gspo import SYSTEM_PROMPT


def _mock_embed(vocab_size=152000, hidden_size=64):
    return nn.Embedding(vocab_size, hidden_size)


def test_cache_template_ids():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    prefix, suffix = cache_template_ids(tok, SYSTEM_PROMPT)
    assert isinstance(prefix, list)
    assert isinstance(suffix, list)
    assert len(prefix) > 5
    assert len(suffix) > 3
    assert all(isinstance(i, int) for i in prefix)
    assert all(isinstance(i, int) for i in suffix)


def test_build_generation_embeds_shape():
    hidden_size = 64
    embed_fn = _mock_embed(hidden_size=hidden_size)
    prefix_ids = [1, 2, 3, 4, 5]
    suffix_ids = [10, 11, 12]
    obs_embeds = torch.randn(14, hidden_size)

    result = build_generation_embeds(prefix_ids, suffix_ids, obs_embeds, embed_fn)
    assert result.shape == (5 + 14 + 3, hidden_size)
    assert result.device == torch.device("cpu")


def test_build_generation_embeds_dtype_matches():
    hidden_size = 64
    embed_fn = _mock_embed(hidden_size=hidden_size)
    prefix_ids = [1, 2, 3]
    suffix_ids = [4, 5]
    obs_embeds = torch.randn(8, hidden_size, dtype=torch.bfloat16)

    result = build_generation_embeds(prefix_ids, suffix_ids, obs_embeds, embed_fn)
    assert result.dtype == torch.bfloat16


def test_build_training_embeds_shape():
    hidden_size = 64
    embed_fn = _mock_embed(hidden_size=hidden_size)
    prefix_ids = [1, 2, 3, 4, 5]
    suffix_ids = [10, 11, 12]
    obs_embeds = torch.randn(14, hidden_size)
    completion_ids = torch.tensor([20, 21, 22, 23])

    result, prompt_len = build_training_embeds(
        prefix_ids, suffix_ids, obs_embeds, completion_ids, embed_fn,
    )
    expected_len = 5 + 14 + 3 + 4
    assert result.shape == (expected_len, hidden_size)
    assert prompt_len == 5 + 14 + 3


def test_build_training_embeds_obs_carries_grad():
    hidden_size = 64
    embed_fn = _mock_embed(hidden_size=hidden_size)
    prefix_ids = [1, 2]
    suffix_ids = [3]
    obs_embeds = torch.randn(4, hidden_size, requires_grad=True)
    completion_ids = torch.tensor([10, 11])

    result, _ = build_training_embeds(
        prefix_ids, suffix_ids, obs_embeds, completion_ids, embed_fn,
    )
    assert result.requires_grad
    result.sum().backward()
    assert obs_embeds.grad is not None
