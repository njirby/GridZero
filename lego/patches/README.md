# Out-of-tree patches for the RL stack

GridZero's RL path runs on lego-rl (`/home/nate/Documents/lego-rl`, upstream
`LegoX/Lego-RL` @ `a3e28f1`) with its vendored verl (`third_party/verl` @ `7aed6b23`).
Neither upstream is ours, so our changes live here instead of being pushed there.

| file | apply in | what |
|---|---|---|
| `lego-rl.patch` | lego-rl root: `git apply lego/patches/lego-rl.patch` | proxy on a fixed UFW-allowed port (8011+), 27B-safe Ray cleanup (+ pipefail fix), LoRA/sdpa/`add_force` hydra args, `AGENT_LOOP_CONFIG_PATH` in oc.env, sync config (`load_format: safetensors`, `layered_summon`), runtime-image pull falls back to the local copy, `HARBOR_MAX_CONCURRENT_TRIALS` per-worker trial cap |
| `agent_loop_config_oc_docker.yaml` | copy to `lego-rl/src/verl_patch/config/` | docker-backend agent loop for the opencode scaffold |
| `verl-7aed6b23.patch` | `third_party/verl`: `git apply` | (superseded by the `-sp` patch) 3D mRoPE nested-tensor rebuild (`tensordict_utils.py`), activation-offload bounds checks |
| `verl-7aed6b23-sp.patch` | `third_party/verl` @ 7aed6b23: `git apply` | everything above + `VERL_ACTIVATION_OFFLOAD_PIN` (unpinned, swappable activation offload) + **upstream verl #6660** (Qwen3.5 linear attention under Ulysses SP via FLA context parallel, packed `cu_seqlens`, fused-loss label fix) + our **sdpa Ulysses all-to-all** for Qwen3.5 full-attention layers (upstream only wires it for flash-attn) + a **training-memory leak fix** in `forward_backward_batch` (stored `model_output` kept autograd graphs alive across micro-batches: +160-500MB per micro-batch until OOM; now detached after backward) + `VERL_MEM_TRACE=1` per-micro-batch GPU memory trace + tests |

Not included: the flash_attn shim inside lego-rl's `.venv` (see `../NOTES.md`), and a
pre-existing unrelated edit to `lego-rl/utils/eval_swerebench_filtered.py`.
Regenerate after changing lego-rl: `git -C <lego-rl> diff -- . ':!utils/eval_swerebench_filtered.py' > lego/patches/lego-rl.patch`.

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
