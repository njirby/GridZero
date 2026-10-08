# Out-of-tree patches for the RL stack

GridZero's RL path runs on lego-rl (`/home/nate/Documents/lego-rl`, upstream
`LegoX/Lego-RL` @ `a3e28f1`) with its vendored verl (`third_party/verl` @ `7aed6b23`).
Neither upstream is ours, so our changes live here instead of being pushed there.

| file | apply in | what |
|---|---|---|
| `lego-rl.patch` | lego-rl root: `git apply lego/patches/lego-rl.patch` | proxy on a fixed UFW-allowed port (8011+), 27B-safe Ray cleanup (+ pipefail fix), LoRA/sdpa/`add_force` hydra args, `AGENT_LOOP_CONFIG_PATH` in oc.env, sync config (`load_format: safetensors`, `layered_summon`), runtime-image pull falls back to the local copy |
| `agent_loop_config_oc_docker.yaml` | copy to `lego-rl/src/verl_patch/config/` | docker-backend agent loop for the opencode scaffold |
| `verl-7aed6b23.patch` | `third_party/verl`: `git apply` | 3D mRoPE nested-tensor rebuild (`tensordict_utils.py`), activation-offload bounds checks |

Not included: the flash_attn shim inside lego-rl's `.venv` (see `../NOTES.md`), and a
pre-existing unrelated edit to `lego-rl/utils/eval_swerebench_filtered.py`.
Regenerate after changing lego-rl: `git -C <lego-rl> diff -- . ':!utils/eval_swerebench_filtered.py' > lego/patches/lego-rl.patch`.
