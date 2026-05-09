# GridZero

> Transformer + GRPO for autonomous power grid control

GridZero trains a language model (Qwen3-0.6B) to control a power grid simulation using **Group Relative Policy Optimization (GRPO)** via [TRL](https://github.com/huggingface/trl). Actions are expressed as structured JSON tool calls decoded with constrained generation (vLLM `structural_tag`) — every model output is guaranteed to be a syntactically valid grid operation.

## Architecture

Two training modes:

**Text-based (default):** Grid observations are serialized as compact JSON and placed in the user message. TRL's `GRPOTrainer` with `environment_factory` handles rollouts end-to-end.

```
grid2op observation  →  JSON text  →  [Qwen3 + vLLM + constrained gen]  →  tool call  →  env.step()
```

**Embedding-based:** A learned encoder (FlatObsEncoder) maps observation vectors into the model's embedding space. Embeddings are injected via vLLM's `EmbedsPrompt` API, and the encoder is trained jointly with the policy via GRPO rewards.

```
grid2op observation  →  FlatObsEncoder  →  obs embeddings ─┐
chat template tokens  →  embed_tokens   →  text embeddings ─┤→ concat → [vLLM EmbedsPrompt] → tool call
```

**Training**: GRPO via TRL's `GRPOTrainer`. For each grid state, G completions are sampled and scored by a composite reward, then a clipped policy gradient update is applied. vLLM runs in colocate mode for generation. No value function or critic needed.

## Project Structure

```
gridzero/
├── env/
│   ├── wrapper.py            make_env() — thin helper over grid2op
│   ├── observation.py        ObsParser: flat vector + PyG graph construction
│   ├── serialization.py      obs_to_prompt: JSON text serialization
│   └── actions.py            Pydantic ToolCall schemas + grid2op action factory
├── encoder/
│   ├── base.py               ObsEncoder abstract base class
│   ├── flat_encoder.py       MLP projection: [flat_dim] → [seq_len, d_model]
│   └── graph_encoder.py      GNN over grid graph (planned)
├── policy/
│   ├── model.py              Qwen3 model init via HuggingFace AutoConfig
│   ├── tokenizer_utils.py    Prompt templates
│   └── logprob.py            Sequence log-prob extraction
├── inference/
│   ├── vllm_client.py        vLLM client with embedding injection
│   └── constrained_gen.py    Constrained generation schema helpers
├── training/
│   ├── gspo.py               GRPO config builder, dataset builder, structural_tag schema
│   ├── env.py                GridEnv — TRL-compatible tool-call environment
│   ├── reward.py             Reward function bridge for TRL GRPOTrainer
│   ├── embedding_trainer.py  EmbeddingGRPOTrainer — obs embedding injection subclass
│   ├── embed_prompt.py       Prompt embedding construction utilities
│   ├── rollout.py            RolloutCollector
│   └── buffer.py             RolloutBuffer
├── rewards/
│   └── grid_rewards.py       Survival, load served, line capacity margin
configs/                      Hydra config tree
scripts/
├── train.py                  Text-based GRPO training entrypoint
├── train_embeds.py           Embedding-based GRPO training entrypoint
├── evaluate.py               Evaluation + metrics
└── serve_vllm.sh             Start vLLM server
tests/                        pytest test suite
```

## Quickstart

```bash
pip install -e ".[dev]"

# Text-based GRPO training (default: Qwen3-4B, configurable)
python scripts/train.py

# Use Qwen3-0.6B for faster iteration
python scripts/train.py policy=qwen3_0_6b

# Embedding-based training (learned observation encoder)
python scripts/train_embeds.py policy=qwen3_0_6b

# Evaluate a checkpoint
python scripts/evaluate.py n_eval_episodes=20
```

## Tests

```bash
# All unit tests (no GPU required)
pytest tests/ -v -k "not smoke"

# GPU integration test (requires CUDA)
CUDA_VISIBLE_DEVICES=0 pytest tests/test_embedding_integration.py -v -m smoke

# Live vLLM schema-constrained generation checks (requires running vLLM server)
pytest -q -m vllm_live tests/test_vllm_schema_live.py
```

## Action Schema

All grid actions are expressed as tool calls with a discriminated-union JSON schema. The schema is enforced at generation time via vLLM `structural_tag` (xgrammar-backed constrained decoding).

| Action            | Key fields                                          |
|-------------------|-----------------------------------------------------|
| `do_nothing`      | —                                                   |
| `set_line_status` | `line_id: int`, `status: "connect"\|"disconnect"`  |
| `change_bus`      | `element_type`, `element_id: int`, `bus: 1\|2`     |
| `redispatch`      | `gen_id: int`, `delta_mw: float`                    |
| `curtail`         | `gen_id: int`, `max_mw: float ≥ 0`                 |
| `storage`         | `storage_id: int`, `mw: float` (+charge, -discharge)|

## Reward

Default composite reward per timestep:

```
r = 1.0 * survival  +  0.5 * load_served_ratio  +  0.2 * line_capacity_margin
```

Blackout (terminal): -1.0. Configure weights in `configs/reward/default.yaml`.

## Why tool calls?

Constrained generation (vLLM structured outputs with JSON schema) guarantees every model output is a syntactically valid grid operation, eliminating invalid-action noise that would otherwise poison the reward signal during early training.

## Roadmap

- [x] Text-based GRPO training with TRL + vLLM colocate
- [x] Learned observation embeddings via FlatObsEncoder + EmbedsPrompt
- [ ] Graph encoder with substation topology (GNN)
- [ ] Multi-step rollouts
- [ ] MuZero world model for lookahead planning

## Dependencies

- [grid2op](https://github.com/rte-france/Grid2Op) + [lightsim2grid](https://github.com/BDonnot/lightsim2grid)
- [TRL](https://github.com/huggingface/trl) >= 0.29 — GRPO via `GRPOTrainer`
- [vLLM](https://github.com/vllm-project/vllm) >= 0.12 — colocate mode generation + `EmbedsPrompt`
- [PyTorch](https://pytorch.org/) + [PyG](https://pyg.org/) (torch-geometric)
- [HuggingFace Transformers](https://github.com/huggingface/transformers) >= 5.2
- [Pydantic](https://docs.pydantic.dev/) v2
- [Hydra](https://hydra.cc/) for config management
