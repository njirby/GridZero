"""Async vllm client for rollout generation with multimodal embedding injection."""
from __future__ import annotations

import asyncio
from typing import Any

import torch
from omegaconf import DictConfig

from gridzero.encoder.embedding_pack import pack_embeddings_for_vllm


class VLLMRolloutClient:
    """Async client for the vllm OpenAI-compatible server.

    Requires vllm to be started with --enable-mm-embeds (see scripts/serve_vllm.sh).
    Encoder embeddings are serialized and passed via multi_modal_data so vllm
    can inject them as prefix tokens before autoregressive generation.
    """

    def __init__(self, cfg: DictConfig) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=cfg.inference.vllm_url,
            api_key=cfg.inference.get("api_key", "gridzero"),
        )
        self.model_name = cfg.inference.model_name
        self.cfg = cfg

    async def sample_completions(
        self,
        embeddings: torch.Tensor,
        prompt: str,
        n: int,
        json_schema: dict,
        max_tokens: int = 256,
        temperature: float = 1.0,
    ) -> list[str]:
        """Sample n tool call completions for one grid state.

        Args:
            embeddings: Encoder output, shape [N, d_model].
            prompt: Formatted prompt string (from tokenizer_utils.build_prompt).
            n: Number of completions to sample (the G in GSPO).
            json_schema: JSON schema dict for vllm guided_json decoding.
            max_tokens: Maximum tokens to generate per completion.
            temperature: Sampling temperature.

        Returns:
            List of n JSON strings, each a valid tool call per the schema.
        """
        mm_data = pack_embeddings_for_vllm(embeddings)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={
                "guided_json": json_schema,
                "multi_modal_data": mm_data,
            },
        )
        return [choice.message.content for choice in response.choices]

    async def health_check(self) -> bool:
        """Return True if the vllm server is reachable."""
        try:
            models = await self.client.models.list()
            return any(m.id == self.model_name for m in models.data)
        except Exception:
            return False
