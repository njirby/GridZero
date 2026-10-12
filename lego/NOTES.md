# GridZero Lego-RL run notes

Non-obvious workarounds and design decisions for the 2×3090 (24GB) GRPO+LoRA smoke
test. These are NOT in the upstream lego-rl/verl docs — future-you (or an agent)
will hit each of these walls. The config is `configs/gridzero_2b_lora.env`.

## Topology: colocated sync on cards 0-1 (`gridzero_2b_lora_scale.env`), or async 2+2 on all four (`gridzero_2b_lora_async.env`, see 2026-10-12)
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

## Findings 2026-10-08 → 10-12 (scale runs, RAM, async 2+2, rollout splitting)
Chronology lives in the session; the load-bearing facts:

**What the policy learned (sync, 7→30 turns, 64 rollouts/step).** With 7 turns it fired several no-op
`simctl act` calls per turn to fast-forward (reward hack; 1249 vs ~125). With do-nothing-relative reward
and 30 turns it converged in ~8 steps to `simctl step && simctl step && …` chains: score ≈ 0 = parity
with do-nothing. Not a hack — the env is easy (do-nothing survives 324/328 scenarios). Real skill needs
harder episodes (follow-up). Side effect: the chains grew to ~200 `simctl` per command (~3.5k tokens of
decode per turn), quadrupling rollout time — rollout speed was token generation, not containers.

**RAM (30 GB box) — read before raising concurrency.**
- CUDA **pinned host memory shows up as `Shmem`** in /proc/meminfo and **cannot be swapped**. The
  "mystery" 18.6 GB of Shmem was our offloads (FSDP param offload ~8 GB fp32 + vLLM sleep weight copy
  4.5 GB + torch's pinned cache). It forced everything else into swap.
- Sustained memory pressure makes **systemd-oomd kill whole tmux scopes** (it killed the user's model
  twice). Guard on PSI (`/proc/pressure/memory` some avg10 > 40% ×3 or avg60 > 30% → stop), not on
  swap size. Swap in use with PSI ≈ 0 is fine.
- Turning off param + activation offload (SP=2 + the leak fix made GPU room) cut Shmem to ~6 GB and
  made steps 29% faster. 32 concurrent trials made each trial 66% slower (paging); 16 is the sweet spot.

**Async 2+2 (`TRAINING_MODE=async`, `gridzero_2b_lora_async.env`).** Trainer (SP=2) on cards 0-1, two
TP=1 vLLM replicas (262k context) on 2-3, staleness 1. Upstream LoRA sync sent base weights only →
fixed to adapter-only (see `patches/README.md`). Verify with `lego/scripts/probe_async_sync.sh`.
- **Keep `ROLLOUT_CORRECTION_BYPASS=False` with GSPO.** GSPO clips the *sequence* ratio at 3e-4/4e-4,
  but vLLM-vs-FSDP numerical mismatch alone is 6e-4 mean / 1.3e-3 max per sequence (measured), so with
  bypass (old = rollout logprobs) noise clips sequences (25% at step 1). Bypass off recomputes old
  logprobs (+17 s forward, ~8 GB more swap from save_model_to_cpu; hidden while the trainer waits).
- Cleanup trap: vLLM rollout processes rename themselves `VLLM::EngineCore` / `VLLM::Worker` and
  survive `pkill -f "[D]ocuments/lego-rl/.venv"`. Kill them by experiment:
  `for p in $(pgrep -f VLLM::); do grep -qz "EXP_NAME=<exp>" /proc/$p/environ && kill -9 $p; done`
  (check `nvidia-smi` is empty afterwards).

**Rollout splitting (train whole long episodes).** Episodes no longer get truncated: opencode compacts
proactively (`OC_CONTEXT_LIMIT`, default off; async config 98304/8192), the proxy archives each context
window as a segment, the agent loop emits one training row per segment, GRPO advantage is shared and
each row weighted 1/N (Agent Lightning / ReSum / verl PR #7580 style). Facts from a real opencode
1.14.22 trace: compaction call system prompt "You are an anchored context summarization assistant…";
resumed thread = `[system, kept tail…, user "What did we do so far?", assistant <summary>, user
<continue>]`; the original task message is dropped and the tail may start with an assistant turn;
the tail is long (22k → 18k tokens) so compaction recurs often. opencode's own prompt is ~12.6k tokens
— a compaction threshold below that loops forever (the scaffold now refuses < 20k).

**Bugs fixed on the way (all in the patches):** lego-rl's verl patches were only half applied (the
trajectory filter never ran); the proxy dropped every post-compaction turn; `_recompute_prefix_logprobs`
raised IndexError on every "MAXIMUM STEPS" history rebuild; vLLM `add_lora` was called per IPC bucket on
reused buffer views.

**End-to-end result (P5, 2026-10-12, `gridzero_2b_lora_async.env`, 4 prompts × 4, compaction at 48k).**
Two updates, exit 0: rows/episodes 18/16 and 17/16 (real compaction splits through opencode → proxy →
agent loop → trainer; the per-episode invariants are asserted in code and held), `pg_clipfrac` 0, trainer
peak 14.7–14.9 GB of 24, PSI 0, weight-sync canary at the numerical floor after every sync. Step time
**384 s vs 16.4 min** for colocated sync (rollouts overlap training; the rollouter never idles, the trainer
waits ~60–70% of the time → rollouts are now the bottleneck). Compaction at 24k is too tight (opencode's
base prompt is ~12.6k tokens); 48k works for probes, the config default is 98304/8192.
- Staleness check: update 2 trained on 8 stale trajectories (sampled under version 0). With bypass off,
  `old_log_prob` is recomputed with the CURRENT weights, so `rollout_corr/kl` jumped 4e-4 → 3.5e-2 = one
  update of real policy drift, previously uncorrected. Fix: `ROLLOUT_IS=token`, threshold 2.0 (truncated
  importance sampling, i.e. decoupled PPO) in the async config. Verified (`logs/p5-is-*.log`, exit 0):
  fresh update — IS mean 1.0001, ESS 0.999, range 0.72–1.31, nothing at the cap; stale update (5 stale
  trajectories) — IS mean 1.0001, ESS 0.997, range 0.055–2.0 (cap reached), 0.03%/0.07% of tokens
  beyond the cap, pg_clipfrac 0. The correction only bites on the tokens that drifted.
- LR 1e-4 on the LoRA moves the policy ~0.035 KL per update and grad_norm rose 0.22 → 0.91; watch it on
  longer runs.


## 2026-10-12 — BUG: task episode params never reached the sim (all runs before this date)
Harbor's docker env in non-compose mode applies task.toml `[environment.env]` only to `docker exec`
calls (`_persistent_env` -> `-e`), never to the container's PID 1. The entrypoint/backend start with
the container, so every task booted the entrypoint defaults: **case14, chronic 0, 24 steps, seed 0**.
Evidence: across 94 case14 task ids in harbor_trials, only 10 distinct t=0 states; the first idf smoke
ran case14 for 24 steps (reward ~57/step; idf is ~194). Meanwhile the verifier subtracted each task's
own chronic's do-nothing baseline, so per-task reward differences were baseline noise, not behavior.
All earlier case14 training effectively saw ONE scenario.
Fix: params now live in each task's `environment/docker-compose.yaml` (Harbor layers it over the
prebuilt compose; checked with Harbor's compose files: PID 1 gets idf/201/288). The entrypoint writes
`/run/gridzero/episode.json` (root-only) and the verifier fails as INFRA_FAILURE unless it matches the
task's baseline row. Old task dirs (`lego/tasks`, `lego/tasks_scale`) predate this: regenerate them
with make_tasks.py before reuse.
