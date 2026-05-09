"""Tests for rollout context plumbing (prompt + embeddings + schema)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


def test_prompt_contains_grid_state_placeholder():
    from gridzero.policy.tokenizer_utils import build_prompt

    prompt = build_prompt()
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "[GRID STATE]" in prompt
    assert "<|im_start|>assistant" in prompt


def test_pack_unpack_embeddings_roundtrip():
    torch = pytest.importorskip("torch")
    from gridzero.encoder.embedding_pack import pack_embeddings_for_vllm, unpack_embeddings_from_vllm

    embeddings = torch.randn(5, 16, dtype=torch.float32)
    payload = pack_embeddings_for_vllm(embeddings)
    restored = unpack_embeddings_from_vllm(payload)
    assert payload["shape"] == [5, 16]
    assert restored.shape == embeddings.shape
    assert torch.allclose(restored, embeddings)


@pytest.mark.asyncio
async def test_vllm_client_sends_multimodal_context(monkeypatch):
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf
    from gridzero.inference.vllm_client import VLLMRolloutClient

    captured: dict = {}

    class _FakeChoice:
        def __init__(self, content: str):
            self.message = type("Message", (), {"content": content})()

    class _FakeResponse:
        def __init__(self):
            self.choices = [_FakeChoice('{"action_type":"do_nothing"}')]

    class _FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = _FakeChat()

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)

    cfg = OmegaConf.create(
        {
            "inference": {
                "vllm_url": "http://localhost:8000/v1",
                "model_name": "Qwen/Qwen3-0.6B",
                "api_key": "gridzero",
                "enable_thinking": False,
            }
        }
    )
    client = VLLMRolloutClient(cfg)
    embeddings = torch.randn(4, 32, dtype=torch.float32)
    prompt = "<|im_start|>user\n[GRID STATE]<|im_end|>\n<|im_start|>assistant\n"
    schema = {"type": "object"}
    out = await client.sample_completions(
        embeddings=embeddings,
        prompt=prompt,
        n=1,
        json_schema=schema,
        max_tokens=32,
        temperature=0.3,
    )
    assert out == ['{"action_type":"do_nothing"}']
    assert captured["messages"][0]["content"] == prompt
    assert captured["extra_body"]["guided_json"] == schema
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    mm = captured["extra_body"]["multi_modal_data"]
    assert mm["type"] == "gridzero_embeddings"
    assert mm["shape"] == [4, 32]


@pytest.mark.asyncio
async def test_rollout_collector_passes_obs_embedding_and_prompt():
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf

    from gridzero.env.observation import ObsData
    from gridzero.training.buffer import RolloutBuffer
    from gridzero.training.rollout import RolloutCollector

    class _FakeObs:
        def __init__(self):
            self.load_p = np.array([1.0], dtype=np.float32)
            self.rho = np.array([0.5], dtype=np.float32)

    class _FakeParser:
        def parse(self, _obs):
            return ObsData(
                flat=np.zeros(8, dtype=np.float32),
                graph=None,
                n_lines=1,
                n_loads=1,
                n_gens=1,
                n_substations=1,
            )

    class _FakeActionSpace:
        n_gen = 1
        n_storage = 0

        def __call__(self, payload):
            return {"payload": payload}

    class _FakeEnv:
        def __init__(self):
            self.action_space = _FakeActionSpace()
            self._done = False

        def reset(self):
            self._done = False
            return _FakeObs()

        def step(self, _action):
            self._done = True
            return _FakeObs(), 0.0, True, {}

        def close(self):
            return None

    @dataclass
    class _FakeEncoder:
        out: "torch.Tensor"

        def __call__(self, _obs_data):
            return self.out

    class _FakeVLLMClient:
        def __init__(self):
            self.calls = []

        async def sample_completions(self, **kwargs):
            self.calls.append(kwargs)
            return ['{"action_type":"do_nothing"}']

    encoder_out = torch.randn(3, 24, dtype=torch.float32)
    encoder = _FakeEncoder(out=encoder_out)
    vllm_client = _FakeVLLMClient()
    cfg = OmegaConf.create(
        {
            "env": {"max_steps": 2},
            "training": {"max_generation_tokens": 16, "temperature": 0.2},
        }
    )
    collector = RolloutCollector(
        encoder=encoder,
        vllm_client=vllm_client,
        json_schema={"type": "object"},
        cfg=cfg,
    )
    buffer = RolloutBuffer(max_size=8)
    prompt = "ctx"

    def _env_factory():
        return _FakeEnv(), _FakeParser()

    episodes = await collector.collect_batch(
        env_factory=_env_factory,
        action_space=_FakeActionSpace(),
        prompt=prompt,
        buffer=buffer,
        n_episodes=1,
    )

    assert len(episodes) == 1
    assert len(vllm_client.calls) == 1
    call = vllm_client.calls[0]
    assert call["prompt"] == prompt
    assert call["n"] == 1
    assert call["json_schema"] == {"type": "object"}
    assert torch.allclose(call["embeddings"], encoder_out)
