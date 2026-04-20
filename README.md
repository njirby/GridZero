# GridZero

> Transformer + GSPO for autonomous power grid control

GridZero trains a randomly-initialized transformer (Qwen3-4B architecture) to control a power grid simulation using **Group Sequence Policy Optimization (GSPO)**. Actions are expressed as structured JSON "tool calls" decoded with constrained generation — every model output is guaranteed to be a syntactically valid grid operation.

## Architecture

```
grid2op observation
        │
        ▼
ObsEncoder  (GNN over grid graph → one embedding per substation)
        │  shape: [N, d_model]
        ▼
vLLM  (--enable-mm-embeds)
        │  encoder embeddings injected as multimodal prefix tokens
        │  guided_json constrains generation to the ToolCall schema
        ▼
JSON tool call  e.g. {"action_type": "set_bus", "element_type": "load", ...}
        │
        ▼
parse_tool_call()  →  grid2op Action
        │
        ▼
env.step(action)  →  reward, next observation
```

**Policy**: Randomly initialized Qwen3-4B architecture. No pretraining — the model builds operational intuition from scratch via environment signals alone.

**Training**: GSPO — for each grid state, G completions are sampled, rewards evaluated, and a sequence-level clipped importance-weighted gradient update is applied. No value function or critic needed.

**Inference**: vLLM hosts the model for rollout efficiency. The graph encoder (PyTorch + PyG) runs separately and injects embeddings via vLLM's multimodal embedding API.

## Project Structure

```
gridzero/
├── env/
│   ├── wrapper.py          make_env() — thin helper over grid2op.gym_compat.GymEnv
│   ├── observation.py      ObsParser: flat vector + PyG graph construction
│   └── actions.py          Pydantic ToolCall schemas + grid2op action factory
├── encoder/
│   ├── graph_encoder.py    GNN over grid graph (primary)
│   └── flat_encoder.py     Linear projection fallback
├── policy/
│   ├── model.py            Randomly-init Qwen3 via HuggingFace AutoConfig
│   ├── tokenizer_utils.py  Prompt templates
│   └── logprob.py          Sequence log-prob extraction for GSPO loss
├── inference/
│   ├── vllm_client.py      Async vLLM client with embedding injection
│   └── constrained_gen.py  guided_json schema builder
├── training/
│   ├── gspo.py             GSPO loss + GSPOTrainer
│   ├── rollout.py          RolloutCollector (async, G completions per state)
│   └── buffer.py           RolloutBuffer
└── rewards/
    └── grid_rewards.py     Survival, load served, line capacity margin
configs/                    Hydra config tree
scripts/
├── train.py                Training entrypoint
├── evaluate.py             Evaluation + metrics
└── serve_vllm.sh           Start vLLM server with --enable-mm-embeds
tests/                      pytest smoke tests
```

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Start vLLM server (after saving a checkpoint, or point at a HF model dir)
GRIDZERO_MODEL_PATH=Qwen/Qwen3-4B bash scripts/serve_vllm.sh

# Train
python scripts/train.py

# Evaluate
python scripts/evaluate.py n_eval_episodes=20
```

## Action Schema

All grid actions are discriminated-union JSON objects. The full schema is auto-generated from Pydantic models and passed to vLLM's `guided_json` at rollout time.

| `action_type`     | Key fields                                          |
|-------------------|-----------------------------------------------------|
| `do_nothing`      | —                                                   |
| `set_line_status` | `line_id: int`, `status: "connect"\|"disconnect"`  |
| `change_bus`      | `element_type`, `element_id: int`, `bus: 1\|2`     |
| `redispatch`      | `gen_id: int`, `delta_mw: float`                    |
| `curtail`         | `gen_id: int`, `max_mw: float ≥ 0`                 |
| `storage`         | `storage_id: int`, `mw: float` (+charge, −discharge)|

## Reward

Default composite reward per timestep:

```
r = 1.0 × survival  +  0.5 × load_served_ratio  +  0.2 × line_capacity_margin
```

Blackout (terminal): `−1.0`. Configure weights in `configs/reward/default.yaml`.

## Why random initialization?

With no prior domain knowledge, any reasoning the model generates is guesswork. RL on guessed reasoning is noisy and unstable. GridZero trains the policy from random weights so it builds genuine grid intuition from environment signals rather than hallucinated priors.

## Why tool calls?

Constrained generation (`guided_json`) guarantees every model output is a syntactically valid grid operation, eliminating a class of invalid-action noise that would otherwise poison the reward signal during early training.

## Roadmap

- [x] Phase 1: GSPO training on single agent, no opponents
- [ ] Phase 2: MuZero world model for lookahead planning
- [ ] Adversarial self-play opponents
- [ ] Natural language chain-of-thought traces (once the model has domain knowledge)
- [ ] Reward signal included in the autoregressive context

## Dependencies

- [grid2op](https://github.com/rte-france/Grid2Op) + [lightsim2grid](https://github.com/BDonnot/lightsim2grid)
- [PyTorch](https://pytorch.org/) + [PyG](https://pyg.org/)
- [vLLM](https://github.com/vllm-project/vllm) ≥ 0.5
- [HuggingFace Transformers](https://github.com/huggingface/transformers) ≥ 4.45
- [Pydantic](https://docs.pydantic.dev/) v2
- [Hydra](https://hydra.cc/) for config management
