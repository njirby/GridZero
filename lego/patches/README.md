# Out-of-tree patches for the RL stack

GridZero's RL path runs on lego-rl (`/home/nate/Documents/lego-rl`, upstream
`LegoX/Lego-RL` @ `a3e28f1`) with its vendored verl (`third_party/verl` @ `7aed6b23`).
Neither upstream is ours, so our changes live here instead of being pushed there.
Both patches below reproduce the full working trees from clean upstream (gated by
applying to a clean checkout and diffing against the live tree).

| file | apply in | what |
|---|---|---|
| `verl-7aed6b23-sp.patch` | `third_party/verl` @ clean `7aed6b23`: `git apply` | **The whole verl delta. Do NOT also apply lego-rl's `patches/verl_*.patch`** — they are merged in (see below). |
| `lego-rl.patch` | lego-rl @ clean `a3e28f1`: `git apply` | The whole lego-rl delta for `scripts/ src/ tests/` (incl. new files: `agent_loop_config_oc_docker.yaml`, compaction fixtures). |
| `agent_loop_config_oc_docker.yaml` | (now inside `lego-rl.patch`) | kept for reference |
| `verl-7aed6b23.patch` | — | superseded by `verl-7aed6b23-sp.patch` |

## What's in `verl-7aed6b23-sp.patch`
- **lego-rl's own verl patches**, which were only half-applied in our tree: `verl_7aed6b23.patch` +
  `verl_sao_7aed6b23.patch` (trajectory filter v6, SAO losses, FSDP1 save/restore, R3). Hand-merged:
  SAO's `validate` kwarg hunk in `agent_loop.py` (a fresh lego-rl `setup_env.sh` is broken on it).
- **Trajectory filter actually wired** and per-episode: drops all rows sharing `episode_key`, fails loud
  on a missing `termination_reason` column or unknown `drop_reasons`. Fully-async no longer forces
  `agent_name`.
- **Sequence parallelism for Qwen3.5**: upstream verl **#6660** (GatedDeltaNet via FLA context
  parallel, packed `cu_seqlens`, fused-loss label fix) + our **sdpa Ulysses all-to-all** for the
  full-attention layers (upstream wires flash-attn only). Verification below.
- **Training-memory leak fix** in `forward_backward_batch` (stored `model_output` kept autograd graphs
  alive across micro-batches, +160–500 MB per micro-batch) + `VERL_MEM_TRACE=1` per-micro-batch trace.
- `VERL_ACTIVATION_OFFLOAD_PIN=0`: unpinned (swappable) activation offload.
- **Async LoRA weight sync (adapter-only)** — upstream sent BASE weights only for unmerged LoRA, so a
  separate vLLM silently kept sampling the base model. Now the trainer sends the adapter
  (`get_per_tensor_param(base_sync_done=True)`, ~64 MB), the `CheckpointEngineManager` passes the
  `peft_config` (fetched once via `get_peft_config()`), and vLLM loads it once per sync (it used to call
  `add_lora` per bucket on views into a reused IPC buffer). Fails loud if a LoRA run syncs without a
  peft config. Opt-in `VERL_SYNC_CANARY=1` scores a fixed prompt on vLLM and the trainer after each sync;
  `[GPU-PLACEMENT]` / `[SYNC-TIME]` / `[LORA-SYNC]` logs.
- **Rollout splitting (multi-row episodes)** — an agent loop may return a list of outputs (one per
  context window). Every row gets `__src_row__`, `episode_key`, `segment_idx`, `n_segments`,
  `loss_weight = 1/N`; the reward sits on one row (episode score = sum). GRPO is computed on the final
  row per episode and broadcast to all its rows (bit-identical to the unsplit episode); batches are
  padded to the DP/mini-batch divisor with inert rows; `agg_loss` takes the per-row weight (policy,
  entropy, KL) with the fixed denominator. No-op when no episode splits. Metrics `rollout/{rows,
  episodes, split_episodes, pad_rows, rows_per_mini_batch}`. Code: `verl/trainer/ppo/split_rollouts.py`.

## What's in `lego-rl.patch`
- Proxy (`vllm_chat_completion_proxy.py`): fixed UFW-allowed port (8011+); **segment archiving** — after
  an opencode compaction the post-compaction main thread is bound back to the episode as a new segment
  (it used to be routed as a sub-agent and silently dropped); `pop_session()` returns `segments`;
  `HARBOR_MAX_SEGMENTS` (default 8); fixed `_recompute_prefix_logprobs` (raised IndexError on every
  prefix-mismatch rebuild, e.g. opencode's "MAXIMUM STEPS" reordering). Markers come from the opencode
  1.14.22 binary; a real compaction trace is a test fixture (`tests/agent_loop/fixtures/`).
- `builtin_swe_agent_loop.py`: one `AgentLoopOutput` per segment; per-worker trial cap
  `HARBOR_MAX_CONCURRENT_TRIALS`.
- `scaffold/oc.env`: opt-in proactive compaction `OC_CONTEXT_LIMIT` / `OC_OUTPUT_LIMIT` (opencode only
  compacts proactively when the model declares `limit.context`; compacts at context − output). Refuses a
  window larger than `MAX_PROMPT + MAX_RESP`, and a threshold < 20k (opencode's own prompt is ~12.6k
  tokens → infinite compaction loop).
- Ray cleanup that spares other vLLM services (+ pipefail fix); LoRA/sdpa/`add_force` hydra args;
  `CHECKPOINT_ENGINE_BUCKET_MB`; preflight `BYPASS_KEEP_LOSS_MODE`; runtime-image pull falls back to the
  local copy; async yaml loads the base from disk (`load_format: safetensors`, `layered_summon`).

## Regenerate (after changing the live trees)
Uses a throwaway index so untracked new files are included and the real index is untouched:

    T=$(mktemp); cd <lego-rl>/third_party/verl
    GIT_INDEX_FILE=$T git read-tree HEAD
    GIT_INDEX_FILE=$T git add -A -- . ':!recipe' ':!*.egg-info' ':!**/__pycache__'
    GIT_INDEX_FILE=$T git diff --cached --binary 7aed6b23 > lego/patches/verl-7aed6b23-sp.patch
    # lego-rl: same, from <lego-rl>, `git add -A -- scripts src tests`, diff against HEAD (a3e28f1)

Gate: `git worktree add --detach <tmp> <base>`, `git apply` the patch there, `diff -r` against the
live tree, `git worktree remove --force <tmp>`.

Not included: the flash_attn shim inside lego-rl's `.venv` (see `../NOTES.md`), and a pre-existing
unrelated edit to `lego-rl/utils/eval_swerebench_filtered.py`.

## Sequence parallelism (SP_SIZE=2) — verify before trusting
Stock verl 7aed6b23 with `SP_SIZE=2` is **silently wrong** for Qwen3.5 (linear-attention layers
restart from zero state per shard; sdpa full attention sees only its own shard). With
`verl-7aed6b23-sp.patch` applied, check equivalence on 2 free GPUs:

    cd third_party/verl && FLA_TILELANG=0 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/special_distributed/test_qwen35_full_model_sp_gridzero.py --models tiny,2b

(`FLA_TILELANG=0`: FLA's TileLang fp32 backward kernel doesn't compile on sm_86; the fp32 pass is
only the test's noise-floor reference.) Result 2026-10-07 on 2×3090, real 2B @ 8k tokens: SP2 vs
SP1 log-prob max/mean 0.287/0.0114 vs bf16-vs-fp32 floor 0.329/0.0117; LoRA grad rel 3.9% vs
floor 4.2%; negative control (sdpa all-to-all removed) 16.4/0.473. ALL PASS.

## Async weight sync — verify
`lego/scripts/probe_async_sync.sh` (2 updates, canary on). 2026-10-12: vLLM moves ~0.04 mean |Δlogprob|
per sync and stays at the vLLM-vs-FSDP base floor (0.027) vs the current trainer; adapter checksum
identical on both replicas; ~2 s per sync after the first.
