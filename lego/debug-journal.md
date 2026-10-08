# GRPO Smoke Test — Debug Journal (2026-10-07)

Companion to `NOTES.md` (which covers the 10-06 fixes: proxy/UFW, flash_attn shim,
layered_summon, Ray memory-monitor, docker reaper). This journal covers the **10-07
session**: getting past the 3D-nested-tensor crash in `compute_log_prob` and then
fighting a series of GPU + CPU memory walls to land the first GRPO step.

**Status at pause (09:31 MDT, superseded by the 16:43 update below):** no run in progress. Last change was
`ENABLE_SLEEP_MODE=False` (keep vLLM weights on-GPU) — **not yet validated**. The
recurring failure is a **CPU-RAM OOM** (kernel SIGKILLs a DataLoader worker), not a
GPU error anymore.

---

## ✅ UPDATE 16:43 — first GRPO step landed (`ENABLE_SLEEP_MODE=False` validated)

Run `2b-lora-20261007-163249`, cards 0-1 only, 27B not running (box rebooted). Exit 0.
- val reward (1 sample) **125.5 → 185.6** after one step; train `critic/score/mean` 73.5, ~8 turns.
- `actor/pg_loss` 1.5e-8, `grad_norm` 0.53, peak GPU alloc 12.1GB / reserved 15.0GB per card.
- Step time ~180s; seqs ~107-119k tokens.
- RAM: ~25GB used + **27GB swap at peak**, during the end-of-run `save_checkpoint`
  (~7.9GB RSS per FSDP WorkerDict). No kernel OOM kill. The
  `DataLoader worker … killed` traceback is printed *after* `Training Progress 100%` —
  it's teardown noise, not the old OOM.
- Only 1 step exists: 3 train rows / batch 3 × `TOTAL_EPOCHS=1`. Raise the epoch count to see a trend.

### 17:00-17:36 — sleep mode back ON (level 1), findings
- `ENABLE_SLEEP_MODE=True` + `GPU_MEM_UTIL=0.7` **OOM'd at vLLM sampler warmup** (needed 970MB, 660MB free):
  `gpu_memory_utilization` is vLLM's own budget and does NOT subtract the FSDP actor already
  resident (4.7GB with `ACTOR_FSDP_PARAM_OFFLOAD=False`). The run then hung with GPUs held — killed.
- `GPU_MEM_UTIL=0.6` started fine: KV cache **314k tokens** (vs 108k at 0.4).
  vLLM log: `sleep freed 9.46 GiB … 2.25 GiB is backed up in CPU and the rest 7.21 GiB is discarded`
  → per card, level-1 sleep copies only the weights (~2.25GB) to RAM; the **KV cache is discarded, not
  offloaded** (vllm/device_allocator/cumem.py `sleep()`). The old "0.6→0.4 saves RAM" comment was wrong.
- verl forces level 1 for LoRA (`vllm_rollout.py:91`, `vllm_async_server.py:943`). Level 2 + vLLM 0.19
  `reload_weights()` from disk would make sleep cost 0 RAM — possible follow-up patch.
- That run never finished: with the user's model back on cards 2-3 (+~8GB RAM), swap hit ~43GB during
  validation (an `opencode-img-shred.py` process at 3.6GB RSS was also resident — not ours), and the
  machine was rebooted at 17:36. **Untested: whether sleep-on survives the training phase's RAM peak.**

Two launch bugs were fixed first (both silent `set -euo pipefail` exits right after `[shm] after cleanup`):
1. `lego-rl/scripts/train/lib/ray.sh`: the 27B-protect `_left=$(pgrep …)` exits 1 when
   no vLLM procs exist (it only ever worked while the 27B was up) → added `|| true`.
2. The reboot renamed the wifi NIC `wlp226s0` → `wlp225s0`; the `ip … dev wlp226s0` pipe failed
   before the `[FATAL]` echo. The configs now derive `TRAIN_NETWORK_INTERFACE` from the default route.

Launcher + monitor now live in `lego/scripts/` (`/tmp/opencode` was wiped by the reboot); logs in `lego/logs/`.

---

## TL;DR — the memory seesaw

The box has **30GB RAM + 2×24GB GPU**. The training wants memory in *both* places at
once, and the multi-turn opencode agent makes sequences long (~150k tokens at 60
turns). Each fix moved the OOM from GPU→CPU or vice-versa:

| # | Symptom | Where | Root cause | Fix | State |
|---|---------|-------|-----------|-----|-------|
| 1 | `size of tensor a (4) must match b (0)` in `to_padded_tensor` | `compute_log_prob` → `chunk_tensordict` | 3D mRoPE `position_ids` nested tensor shape metadata inconsistent | `as_nested_tensor` + `rebuild_3d_nested_from_values` | ✅ fixed |
| 2 | CUDA OOM `18.53 GiB` at `lm_head` | `compute_log_prob` | full-vocab logits for ~61k tokens | `FUSED_KERNELS=True` (chunked `FusedLinearForPPO`) | ✅ fixed |
| 3 | CUDA OOM `456 MiB` in `fla/gated_delta_rule` | validation forward | ~16GB activations for long seq | tried activation offload → | ⚠️ see 4 |
| 4 | `KeyError: 51` in `activation_offload.py:330` | `update_actor` | offload window map mis-sized for Qwen3.5's ~52 FSDP units | bounds checks (fwd+ bwd) | ✅ fixed (but offload later disabled) |
| 5 | `DataLoader worker … killed by signal` (SIGKILL) | training | **CPU RAM** OOM: 150k-token seqs × offloads > 30GB | turn cap + `GPU_MEM_UTIL` + sleep-mode off | 🔁 thrashing |
| 6 | `Available KV cache memory: -1.04 GiB` | vLLM init | `GPU_MEM_UTIL=0.25` too small for the 4GB model | back to 0.4 | ✅ fixed |

The net: **GPU is solved; CPU RAM is the remaining wall.**

---

## 1. The 3D nested-tensor crash (fixed)

**Symptom:** `compute_log_prob` → `prepare_micro_batches` → `chunk_tensordict` →
`to_padded_tensor` → `RuntimeError: The size of tensor a (4) must match the size of
tensor b (0) at non-singleton dimension 0`.

**What's special:** Qwen3.5-2B is a **hybrid** model — 24 layers, mostly
`linear_attention` with `full_attention` every 4th (6 total), `hidden_size=2048`,
vocab ≈ **151936**, plus 1 MTP layer. Its `position_ids` is a **3D** mRoPE nested
tensor `(B, 4, seq_len)` — 4 position tracks × seq_len, ragged on the last dim.

**Two distinct sub-bugs, both in `third_party/verl/verl/utils/tensordict_utils.py`:**

1. **`nested_tensor_from_tensor_list`** built the batch with
   `torch.nested.nested_tensor_from_jagged(values, offsets)` + a manual
   `_ragged_idx` overwrite. For 2D per-sample values that **misplaces the jagged
   marker** (shape reports `(B, j, total)` instead of `(B, 4, j)`) and breaks
   `to_padded_tensor`. Verified in isolation: `nested_tensor_from_jagged` → shape
   `(12, j1, 119)`, `to_padded → (12,4,4)` (wrong); `as_nested_tensor` → shape
   `(12, 4, j2)`, `to_padded → (12,4,20)` (right). **Fix:** when the ragged dim is the
   last dim, use `torch.nested.as_nested_tensor(tensors, layout=torch.jagged)`.

2. **Ray pickle/unpickle + `consolidate()`** corrupts the 3D nested tensor's shape
   metadata on transfer (jagged marker jumps dim 2 → dim 1; `_ragged_idx` 2 → 1), even
   though `values` + `offsets` stay intact. `maybe_fix_3d_position_ids` only reset
   `_ragged_idx`, which isn't enough. **Fix:** new helper
   `rebuild_3d_nested_from_values(nt)` that reconstructs a well-formed tensor from
   `values`+`offsets` (handles both `(B,rope,seq)` and `(B,seq,rope)` layouts). Called
   from `chunk_tensordict` (before `unbind`) and refactored into
   `maybe_fix_3d_position_ids`. Also fixed the `chunk_tensordict` fallback slicing,
   which indexed the wrong dim for 3D (`pc[j, :seq]` → `pc[j, :, :seq]`).

**Validation:** an e2e unit test (`/tmp/opencode/test_e2e.py`) chunking a 3D tensor
twice, each level preceded by a pickle round-trip — all chunks come out `(B,4,j)
ri=2`, `to_padded` correct, data intact. Then the real run got *past* `compute_log_prob`.

---

## 2. `lm_head` CUDA OOM (fixed)

**Symptom:** `Tried to allocate 18.53 GiB` at `logits = self.lm_head(hidden_states)`
(`qwen3_5.py:196`), 14.6GB free.

**Root cause:** the normal backend materializes the full `(T, vocab)` logits. For
~61k tokens: `61000 × 151936 × 2 bytes ≈ 18.5GB`. Even a single 80k-token sequence is
~24GB — bigger than the card.

**Fix:** `FUSED_KERNELS=True` → `impl_backend=torch` → `forward_with_torch_backend`
→ `FusedLinearForPPO`, which computes `log_probs`/`entropy` in **512-token chunks**
without materializing logits (peak ~300MB). 
**Caveat (accepted for smoke test):** it reads `self.lm_head.weight` (base weight only),
ignoring the LoRA delta — exact at step 0 (LoRA B is zero-init), slightly off later.

---

## 3+4. FLA linear-attention OOM → activation offload → `KeyError: 51`

**Symptom 3:** `Tried to allocate 456 MiB` in `fla/ops/gated_delta_rule/chunk.py`
during **validation** forward. The linear-attention (hybrid) layers + ~16GB of
activations for the long sequence didn't fit beside the asleep-vLLM residual (~3GB).

**Attempted fix:** `ENABLE_ACTIVATION_OFFLOAD=True`. It cleared the GPU OOM and reached
`update_actor`, then hit:

**Symptom 4:** `KeyError: 51` in `activation_offload.py:330`
(`synchronize_on_group_commit_forward`: `self.layer_window_map[self.offloaded_group_count]`).

**Root cause:** `get_layers` counts **~52** FSDP-wrapped units for Qwen3.5 (so
`num_offload_group = 51`), but the offload window map is built from a different commit
count, so `offloaded_group_count` increments to 51 — one past the last valid key.

**Fix:** bounds checks on the index (forward line ~330 and backward line ~381) so an
out-of-range count is skipped instead of KeyError'ing. (Activation offload was later
**disabled again** in #5 because it moved the OOM to CPU RAM.)

---

## 5. CPU-RAM OOM — the current wall (thrashing)

**Symptom:** `RuntimeError: DataLoader worker (pid …) is killed by signal: Killed`
(SIGKILL = the OS OOM killer). No GPU error.

**Root cause (confirmed with a per-process memory monitor):** the box peaks at
**~27GB RAM + ~37GB swap ≈ 57-64GB total** vs 30GB physical. The hogs:

| Process | CPU RSS | Why |
|---------|---------|-----|
| `VLLM::Worker_TP0/TP1` | ~4.5GB each (**~9GB**) | vLLM **sleep mode** offloads the 2B weights to RAM |
| `ray::AgentLoopWorker` ×N | ~1.2-1.3GB each | in-container opencode agents |
| `ray::WorkerDict.*` | ~1.2-2.3GB each | FSDP workers; ref model `param_offload=True` on CPU |
| 27B (`vllm serve`, cards 2-3) | ~4GB | me |
| opencode | ~1GB | this CLI |

The long sequences (~150k tokens at 60 turns) are what make both the GPU activations
and the CPU offloads big.

**Lever pull (chronological):**
- `HARBOR_AGENT_MAX_ITERATIONS` 60 → 30 (seqs still ~150k) → 24 (seqs ~138k, GPU OOM
  100MB short) → 22 → **7** (smoke-test compromise; real episodes are longer).
- `ENABLE_ACTIVATION_OFFLOAD` True → **False** (it moved the OOM to CPU RAM).
- `GPU_MEM_UTIL` 0.6 → 0.4 (shrink vLLM KV cache offloaded to CPU). Tried 0.25 →
  `Available KV cache memory: -1.04 GiB` (model doesn't fit) → back to **0.4**. KV
  cache is only **2.5GB / 108,800 tokens** at 0.4, so this lever is small.
- `ACTOR_FSDP_PARAM_OFFLOAD` True → **False** (keep actor weights on-GPU, save ~4GB CPU;
  ref stays offloaded).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → **reverted**: vLLM's
  `CuMemAllocator` (sleep mode) asserts `expandable_segments:True not in ...`.
- **`ENABLE_SLEEP_MODE` True → False** (latest, *unvalidated*): keep vLLM weights on-GPU
  instead of offloading ~9GB to RAM. GPU has headroom now (short seqs): vLLM ~6.5GB +
  FSDP ~10GB ≈ 16.5GB < 24GB.

**If still OOM:** `AGENT_NUM_WORKERS` 4 → 2 (drop ~3GB of AgentLoop workers), and/or
`REF_FSDP_PARAM_OFFLOAD` → False (drop ~4GB, costs GPU).

---

## Current config (`configs/gridzero_2b_lora.env`)

```
SP_SIZE=1
GEN_TP=2
ENABLE_SLEEP_MODE=False        # NEW (unvalidated) — keep vLLM weights on-GPU
GPU_MEM_UTIL=0.4               # was 0.6
MAX_PROMPT=49152
MAX_RESP=147456
USE_DYNAMIC_BSZ=False
FUSED_KERNELS=True             # NEW — chunked log-prob, avoids full-vocab logits
ACTOR_FSDP_PARAM_OFFLOAD=False # was True
ACTOR_FSDP_OPTIMIZER_OFFLOAD=True
REF_FSDP_PARAM_OFFLOAD=True
ENABLE_ACTIVATION_OFFLOAD=False
HARBOR_AGENT_MAX_ITERATIONS=7  # was 60 — smoke-test cap
```

## Code changes (10-07)

`third_party/verl/verl/utils/tensordict_utils.py`
- `nested_tensor_from_tensor_list`: use `as_nested_tensor` when ragged dim is last.
- `rebuild_3d_nested_from_values(nt)`: new helper (rebuild 3D nested from values+offsets).
- `chunk_tensordict`: rebuild 3D nested tensors before `unbind`; fixed 3D fallback slicing.
- `maybe_fix_3d_position_ids`: uses the rebuild helper.

`third_party/verl/verl/utils/activation_offload.py`
- bounds checks at the `layer_window_map[...]` index (forward + backward).

---

## The fundamental tension

30GB RAM is the binding constraint, and it's shared by: the 27B (~4GB), opencode
(~1GB), Ray control (~2GB), the FSDP ref model (offloaded ~4GB), the opencode agent
workers (~5-6GB), and — the big one — the vLLM weights when asleep (~9GB). The long
multi-turn agent episodes are what inflate the GPU activations *and* the CPU offloads
simultaneously. There is no single config that fits everything comfortably; we're
balancing GPU (fused kernels + short seqs) against CPU (sleep mode + offloads).

## Next steps (on resume)

1. **Validate `ENABLE_SLEEP_MODE=False`** — relaunch, watch the mem monitor. Goal: peak
   RAM < 30GB (no swap thrash), reach `actor/pg_loss=`.
2. If still CPU-OOM: `AGENT_NUM_WORKERS=4→2`, then `REF_FSDP_PARAM_OFFLOAD=False`.
3. First `actor/pg_loss=` = GRPO step landed. Then watch the reward trend across steps.
4. **When it works:** raise `HARBOR_AGENT_MAX_ITERATIONS` back up (7 is a smoke cap) and
   re-balance memory; consider sequence parallelism (`SP_SIZE=2`) if 150k-token episodes
   must be kept.
5. Cleanup: remove any leftover debug prints; fold the confirmed fixes into `NOTES.md`.

## Commands

```bash
# launch (cgroup 16G/12G, RAY_memory_monitor off)
setsid tmux new-session -d -s grz-train "sudo -n bash /tmp/opencode/run_train_cgroup.sh \
  > /tmp/opencode/grz-train.log 2>&1; echo EXIT=\$? >> /tmp/opencode/grz-train.log" \
  </dev/null >/dev/null 2>&1

# per-process memory monitor (RAM/swap + top-8 RSS, 15s) → /tmp/opencode/mem.log
bash /tmp/opencode/mem_monitor.sh

# cleanup before each launch
pkill -9 -f "[D]ocuments/lego-rl/.venv"
for c in $(docker ps -aq --filter name=oc-gridzero); do docker rm -f "$c"; done
for n in $(docker network ls --format '{{.Name}}' | grep oc-gridzero); do docker network rm "$n"; done
HOME=/home/nate bash /home/nate/Documents/GridZero/lego/base/build.sh   # rebuild base if pruned

# NEVER: pkill -f vllm   (matches the 27B = the assistant)
```
