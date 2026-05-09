"""Validate GRPO learning: baseline eval -> train -> checkpoint evals -> learning curve."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer

from gridzero.env.actions import parse_tool_call
from gridzero.env.serialization import obs_to_prompt
from gridzero.env.wrapper import make_env
from gridzero.inference.constrained_gen import build_outlines_generator
from functools import partial

from gridzero.training.env import GridEnv
from gridzero.training.gspo import build_dataset, build_grpo_config
from gridzero.training.reward import grid_reward


def _is_peft_checkpoint(path: str) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def evaluate_checkpoint(
    checkpoint_path: str,
    cfg: DictConfig,
    n_episodes: int,
    base_model_name: str = "Qwen/Qwen3-0.6B",
    max_new_tokens: int = 96,
    temperature: float = 0.7,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if _is_peft_checkpoint(checkpoint_path):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    generator = build_outlines_generator(model, tokenizer)

    env, obs_parser = make_env(cfg)
    episode_metrics: list[dict] = []
    total_parse_fail = 0
    total_actions = 0

    try:
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            total_steps = 0

            while not done and total_steps < cfg.env.max_steps:
                obs_data = obs_parser.parse(obs)
                prompt = obs_to_prompt(obs_data)
                total_actions += 1

                json_blob = generator(
                    prompt, temperature=temperature, max_new_tokens=max_new_tokens,
                )

                try:
                    action = parse_tool_call(json_blob, env.action_space)
                except Exception:
                    action = env.action_space({})
                    total_parse_fail += 1

                obs, reward, done, _ = env.step(action)
                total_reward += float(reward)
                total_steps += 1

            episode_metrics.append({
                "episode": ep,
                "steps": total_steps,
                "reward": total_reward,
                "survived": (not done) and total_steps >= cfg.env.max_steps,
            })
    finally:
        env.close()

    del model, tokenizer, generator
    gc.collect()
    torch.cuda.empty_cache()

    survival_rate = sum(int(m["survived"]) for m in episode_metrics) / max(1, len(episode_metrics))
    mean_steps = sum(m["steps"] for m in episode_metrics) / max(1, len(episode_metrics))
    mean_reward = sum(m["reward"] for m in episode_metrics) / max(1, len(episode_metrics))
    parse_fail_rate = total_parse_fail / max(1, total_actions)

    return {
        "survival_rate": survival_rate,
        "mean_steps": mean_steps,
        "mean_reward": mean_reward,
        "parse_fail_rate": parse_fail_rate,
        "episodes": episode_metrics,
    }


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    val_cfg = cfg.get("validation", {})
    if val_cfg.get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("Validation requires CUDA GPU.")

    model_name = val_cfg.get("model_name", "Qwen/Qwen3-0.6B")
    max_steps = int(val_cfg.get("max_steps", 100))
    n_eval_episodes = int(val_cfg.get("n_eval_episodes", 5))
    save_steps = int(val_cfg.get("save_steps", max(1, max_steps // 4)))
    dataset_size = int(val_cfg.get("dataset_size", 32))
    output_dir = str(Path(cfg.output_dir) / "validation")

    # Step 1: Baseline eval
    print(f"\n=== Step 1: Baseline eval ({n_eval_episodes} episodes, model={model_name}) ===")
    baseline = evaluate_checkpoint(model_name, cfg, n_eval_episodes, base_model_name=model_name)
    print(f"  mean_reward   : {baseline['mean_reward']:.4f}")
    print(f"  mean_steps    : {baseline['mean_steps']:.1f} / {cfg.env.max_steps}")
    print(f"  survival_rate : {baseline['survival_rate']:.2%}")
    print(f"  parse_fail    : {baseline['parse_fail_rate']:.1%}")

    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Train
    print(f"\n=== Step 2: GRPO training ({max_steps} steps, save every {save_steps}) ===")
    grpo_cfg = build_grpo_config(cfg, output_dir=output_dir)
    grpo_cfg.max_steps = max_steps
    grpo_cfg.save_steps = save_steps

    dataset = build_dataset(cfg)
    if dataset_size:
        dataset = dataset.select(range(min(dataset_size, len(dataset))))

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=grid_reward,
        train_dataset=dataset,
        environment_factory=partial(GridEnv, env_name=cfg.env.env_name),
        args=grpo_cfg,
    )
    trainer.train()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Evaluate checkpoints
    print(f"\n=== Step 3: Evaluate checkpoints ===")
    checkpoint_dirs = sorted(
        [p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[1]),
    )

    results = [{"step": 0, "checkpoint": model_name, **baseline}]

    for ckpt_dir in checkpoint_dirs:
        step = int(ckpt_dir.name.split("-")[1])
        print(f"\n  Evaluating checkpoint-{step} ({n_eval_episodes} episodes)...")
        metrics = evaluate_checkpoint(
            str(ckpt_dir), cfg, n_eval_episodes, base_model_name=model_name,
        )
        results.append({"step": step, "checkpoint": str(ckpt_dir), **metrics})
        print(f"    mean_reward   : {metrics['mean_reward']:.4f}")
        print(f"    mean_steps    : {metrics['mean_steps']:.1f}")
        print(f"    survival_rate : {metrics['survival_rate']:.2%}")
        print(f"    parse_fail    : {metrics['parse_fail_rate']:.1%}")

    # Step 4: Learning curve
    print(f"\n=== Learning Curve ===")
    print(f"{'Step':>6} | {'Reward':>10} | {'Steps':>8} | {'Survival':>10} | {'Parse Fail':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['step']:>6} | {r['mean_reward']:>10.4f} | "
            f"{r['mean_steps']:>8.1f} | {r['survival_rate']:>9.2%} | "
            f"{r['parse_fail_rate']:>9.1%}"
        )

    # Save results
    results_path = Path(output_dir) / "validation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [{k: v for k, v in r.items() if k != "episodes"} for r in results]
    results_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")

    # Verdict
    final = results[-1]
    improved = final["mean_reward"] > baseline["mean_reward"]
    print(f"\n=== Verdict ===")
    print(f"  Baseline reward : {baseline['mean_reward']:.4f}")
    print(f"  Final reward    : {final['mean_reward']:.4f}")
    print(f"  Delta           : {final['mean_reward'] - baseline['mean_reward']:+.4f}")
    print(f"  Learning signal : {'YES' if improved else 'NO'}")

    if not improved:
        print("\n  WARNING: No improvement detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
