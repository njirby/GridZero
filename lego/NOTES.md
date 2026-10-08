# GridZero Lego-RL run notes

Non-obvious workarounds and design decisions for the 2×3090 (24GB) GRPO+LoRA smoke
test. These are NOT in the upstream lego-rl/verl docs — future-you (or an agent)
will hit each of these walls. The config is `configs/gridzero_2b_lora.env`.

## Topology: 2-GPU colocated (cards 0-1); cards 2-3 belong to the user's own model
- This box: 4×3090, **30GB RAM** + 71GB swap. The 27B (me) runs on cards 2-3 via
  `vllm serve` (port 8001, venv `~/ml/venvs/vllm-qwen35`). RL training uses cards 0-1.
- `NGPUS_PER_NODE=2`, `GEN_TP=2` (ONE vLLM replica, TP=2 across both cards — not two
  single-GPU replicas; a 2nd full model offloaded to CPU on sleep busts the 30GB RAM).
- FSDP DP=2 + vLLM TP=2 share the cards via verl sleep mode: during rollout vLLM is
  awake/FSDP offloaded; during train FSDP is awake/vLLM asleep; the only overlap is the
  ~1s weight-sync handoff.

## CPU OOM → swap fix (the "why OOM with 70GB swap?" answer)
- Ray's raylet has a **memory monitor** that preemptively kills a worker at
  `RAY_memory_usage_threshold` (0.95) of RAM — BEFORE the OS ever touches swap. It only
  watches RAM, never swap, so the 71GB swap sat unused.
- Fix: `RAY_memory_monitor_refresh_ms=0` (disable the preemptive killer) so the kernel
  swaps instead. Set in the launcher `run_train_cgroup.sh` (it must reach the *raylet*
  env). **Gotcha:** if a stale Ray head is still up, a new run attaches to it and the
  env var never lands — always fully kill ray (`ray stop --force` + `pkill` the lego-rl
  venv procs) before relaunching, then verify the raylet's `/proc/<pid>/environ`.
- **Never `pkill -f vllm`** — that also matches the 27B (`vllm serve`) and kills the
  assistant. Kill by the lego-rl venv path (`pkill -f "[D]ocuments/lego-rl/.venv"`).

## GPU OOM at weight-sync → layered_summon + load_format=safetensors
- With `load_format=dummy`, vLLM holds dummy weights so FSDP must stream the **whole**
  model; the all-gather spikes each 24GB card to ~18GB (FSDP) + ~5GB (vLLM) → CUDA OOM.
- Fix in `src/verl_patch/config/lego_rl_sync.yaml` (`actor_rollout_ref.rollout`):
  - `load_format: safetensors` — vLLM loads the real base from disk; FSDP only syncs the
    small LoRA delta.
  - `layered_summon: true` — all-gather the LoRA delta layer-by-layer instead of the whole
    model, keeping FSDP at its ~9GB baseline. `layered_summon` REQUIRES `safetensors`.

## flash_attn shim (no compiled flash-attn installed; 3 consumers probe it)
- verl imports `flash_attn.bert_padding` (pure-torch padding utils) unconditionally in
  `_compute_old_log_prob → unpad_input`. vLLM, if `find_spec("flash_attn")` is present,
  imports `flash_attn.ops.triton.rotary`. transformers checks `is_flash_attn_2_available()`.
- We don't have the compiled package (building on a 30GB box is slow/OOM-prone) and the
  model uses SDPA, so we ship a **shim** in
  `.venv/lib/python3.12/site-packages/flash_attn/`:
  - `bert_padding.py` — pure-torch port of the 4 padding fns (index_first_axis, pad_input,
    unpad_input, + re-export `rearrange` from einops).
  - `ops/triton/rotary.py` — re-exports `apply_rotary` from vLLM's own bundled copy
    (`vllm.vllm_flash_attn.ops.triton.rotary`).
  - **`flash_attn-2.0.0.dist-info/`** (METADATA + top_level.txt + RECORD) — makes it a real
    *distribution* so `importlib.metadata.packages_distributions()` maps it (without this,
    transformers `KeyError: 'flash_attn'`). Version is **2.0.0, below the 2.3.3 gate**, so
    `is_flash_attn_2_available()` returns False (transformers won't try to use FA kernels
    the shim lacks) while `flash_attn.bert_padding` stays importable.

## Agent-loop env var name
- `lego_rl_sync.yaml` + `scaffold/oc.env` read **`AGENT_LOOP_CONFIG_PATH`** (not
  `AGENT_LOOP_CONFIG`). The config must set `AGENT_LOOP_CONFIG_PATH=.../agent_loop_config_oc_docker.yaml`
  or it falls back to `agent_loop_config_oh.yaml` (KubernetesEnvironment) → k8s client
  errors + "no verifier result".

## Rollout proxy must be on a UFW-allowed host port (the big one)
- Each AgentLoopWorker runs an in-process LLM proxy (`vllm_chat_completion_proxy.py`) that
  the in-container opencode agent calls (`{env:HOSTED_VLLM_BASE_URL}`). The container →
  host path is **UFW**: `Default: deny (routed)`, and the only docker→host rules are
  `8011/tcp` + `8011:8020/tcp ALLOW IN 172.16.0.0/12`. Everything else (e.g. 8001) times out.
- The proxy originally bound `port=0` (random ephemeral) → not UFW-allowed → every LLM call
  hung → 30-min trial timeouts → empty trajectories → `compute_log_prob` crash.
- Fix: proxy now binds a **fixed base port** `HARBOR_PROXY_BASE_PORT` (default 8011) with
  increment-on-EADDRINUSE (4 workers → 8011-8014). UFW allows 8011:8020. Advertise host is
  the ray node IP (192.168.0.99). Containers reach it (verified: 8011 = REFUSED not TIMEOUT).

## Docker base image (now `v2`, hardened) + the reaper
- Trials run in containers from **`gridzero-rl-base:v2`** (set in `task_template/task.toml`
  `docker_image` and the stub `task_template/environment/Dockerfile`). `v1` is the
  pre-hardening image, kept for rollback (switch the two references back + rerun make_tasks).
- Full build: `bash lego/base/build.sh gridzero-rl-base:v2` (needs `$HOME/data_grid2op` and
  PyPI). **When PyPI is slow/unreachable** (it was on 10-07: ~0 B/s), layer only the changed
  files onto an existing image — seconds, no network:
  `UPDATE_FROM=gridzero-rl-base:v1 bash lego/base/build.sh gridzero-rl-base:v2`
  (`lego/base/Dockerfile.update` must stay in sync with the tail of `Dockerfile`).
- **A `daw-farm/reaper` may prune images** after trials stop; `scripts/run_train.sh` rebuilds
  `v2` if it's missing. Regenerate tasks after any template change:
  `.venv/bin/python lego/make_tasks.py --out lego/tasks --chronic 0-3 --horizon 24 --seed 0`.

## Launching
- `bash lego/scripts/run_train.sh [config.env]` (logs → `lego/logs/`); `lego/scripts/mem_monitor.sh`
  logs RAM/swap/GPU every 15 s. The launcher disables Ray's memory monitor, pins
  `CUDA_VISIBLE_DEVICES=0,1`, and refuses to start if a stale lego-rl Ray is running.
- Cleanup between runs: `ray stop --force; pkill -9 -f "[D]ocuments/lego-rl/.venv"`, then
  `docker rm -f` the `oc-gridzero*` containers and networks. **Never `pkill -f vllm`.**
- Our lego-rl + verl changes are exported in `lego/patches/` (see its README); lego-rl's own
  origin is upstream, so they are not pushed there.

## Findings from 2026-10-07 (first GRPO steps landed)
Full chronology: `debug-journal.md`. The load-bearing facts:

**Launch-time traps**
- `ray.sh` cleanup aborted silently under `set -euo pipefail` when no vLLM process existed
  (`_left=$(pgrep ...)` exits 1) — only worked while the old 27B was up. Fixed with `|| true`.
- The wifi NIC renames across reboots (`wlp226s0` → `wlp225s0`); a hard-coded
  `TRAIN_NETWORK_INTERFACE` made `ip ... dev` fail before the `[FATAL]` echo. Configs now use
  `$(ip route show default | awk '{print $5; exit}')`.
- lego-rl `remote_docker.py` ran `docker pull` on the opencode runtime image before each run
  and gave up on failure → no runtime mounted → trials looped `[OC-DETECT] fail attempt=30/30`
  then fell back to `apt-get install` (GPUs loaded but 0% util). Now falls back to the
  local image (`using local copy of ...`).

**GPU memory / sleep mode** (the `GPU_MEM_UTIL` knob)
- vLLM `sleep()` level 1 (verl forces level 1 for LoRA) copies only the *weights* to RAM
  (~2.25 GB per TP rank) and **discards the KV cache** (`vllm/device_allocator/cumem.py`).
  So `GPU_MEM_UTIL` costs no RAM; the old "0.6→0.4 to save RAM" reasoning was wrong.
- `gpu_memory_utilization` is vLLM's *own* budget — it does not subtract the FSDP actor
  already resident during rollout (4.7 GB with `ACTOR_FSDP_PARAM_OFFLOAD=False`).
- vLLM warms up its sampler with `max_num_seqs` (default 1024) dummy requests; that ~1 GB
  spike OOM'd 0.7. `ROLLOUT_MAX_NUM_SEQS=64` removes it.
- Result at `GPU_MEM_UTIL=0.7`, `ROLLOUT_MAX_NUM_SEQS=64`, sleep on: KV cache **562k tokens**
  (was 108k at 0.4/sleep off); sleep hands training **15.1 GiB/card**. Going higher (~0.85)
  needs `ACTOR_FSDP_PARAM_OFFLOAD=True` (~4-5 GB more RAM).
- Training phase (12 rollouts × ~110k tokens, 7 turns): 14.3 GB reserved/card; RAM peak
  24 GB + 37 GB swap with the user's model running on cards 2-3. Slow swap is acceptable.
- The `DataLoader worker ... killed by signal` traceback appears AFTER `Training Progress
  100%` — teardown noise, not an OOM (dmesg shows no OOM kill).

**Reward integrity (task hardening)** — previously the model could forge its reward:
- No `SIM_API_TOKEN` was set in the image, so `/sim/reset`, `/sim/step {"n":50}`,
  `/bench/start` were open; the agent ran as root next to the backend + verifier.
- Now: the entrypoint generates a random token passed only to uvicorn (root-only copy in
  `/run/gridzero/token`); `SIMCTL_NO_RESET=1`; `[agent] user = "agent"` (uid 1000);
  backend bound to 127.0.0.1. Verified in a live container as `agent`: privileged routes
  403, n=50 advances 1 step, token/environ unreadable, backend unkillable, port 8731
  can't be re-bound, `/app` read-only, backend unreachable off-loopback.
- `test.sh` writes no reward on a dead backend (exit 1, `/logs/verifier/INFRA_FAILURE`) →
  Harbor retries the trial (bounded: `HARBOR_MAX_RETRIES=2`). Caveat: after the last retry
  lego-rl still scores 0.0 rather than dropping it (needs a lego-rl change).
- Still open: with few turns, reward ≈ steps survived; a `for i in ...; do simctl step; done`
  loop in one bash call is legal and would beat honest play once turns are plentiful.

**Model-facing correctness** (these shaped what the policy learns)
- `render` was disabled in the image but AGENTS.md made it command #3 → 3-4 of 7 turns
  wasted on a traceback; 6/20 rollouts never stepped. Now optional, clean error.
- `simctl step` printed a fake `overloads=[]`; illegal actions gave no reason; after
  game-over the backend said "run `simctl reset`"; `info["disc_lines"]` (a per-line cascade-
  level array) was decoded as line ids → the model was told the wrong line tripped;
  `obs.simulate()` mutated the action in place so out-of-ramp redispatch was clipped and
  scored as legal (+62) instead of `Ambiguous action` (0). All fixed with regression tests.
- Measured on grid2op 1.12.4: a line >100% trips on its 3rd consecutive overloaded step;
  tripped lines are down for 10 steps; redispatch limits per step ±5/±10/±15 MW
  (gen_1_0 / gen_2_1 / gen_0_5).

**Results**: run `2b-lora-20261007-195243` (sleep on, v2 image, non-root agent): exit 0,
`critic/score/mean` 104.4 (12 rollouts), `pg_loss`≈0, `grad_norm` 0.58, step 198 s.
