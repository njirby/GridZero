"""Evaluation script: run grid2op episodes with a trained checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from gridzero.env.actions import parse_tool_call
from gridzero.env.serialization import obs_to_prompt
from gridzero.env.wrapper import make_env
from gridzero.inference.constrained_gen import build_outlines_generator


def _extract_json_blob(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _resolve_checkpoint_path(output_dir: str) -> str:
    output = Path(output_dir)
    if (output / "config.json").exists():
        return str(output)

    checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        return str(checkpoints[-1])
    return str(output)


def _run_constrained_generation(
    generator, prompt: str, temperature: float, max_new_tokens: int,
) -> str:
    return generator(prompt, temperature=temperature, max_new_tokens=max_new_tokens)


def _run_unconstrained_generation(
    model, tokenizer, prompt: str, device: str, max_new_tokens: int, temperature: float,
) -> str | None:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
    return _extract_json_blob(completion)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Load a checkpoint and evaluate policy behavior on grid2op."""
    torch.manual_seed(cfg.seed)

    checkpoint_path = cfg.get("checkpoint_path") or _resolve_checkpoint_path(cfg.output_dir)
    tokenizer_name = cfg.get("tokenizer_name") or checkpoint_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    use_constrained = bool(cfg.get("eval", {}).get("constrained", True))
    generator = None
    if use_constrained:
        generator = build_outlines_generator(model, tokenizer)
        print("Eval using constrained generation (outlines regex)")

    env, obs_parser = make_env(cfg)
    n_eval = int(cfg.get("n_eval_episodes", 1))
    max_new_tokens = int(cfg.get("eval", {}).get("max_new_tokens", 96))
    temperature = float(cfg.get("eval", {}).get("temperature", 0.7))

    episode_metrics: list[dict] = []
    total_parse_fail = 0
    total_actions = 0

    try:
        for ep in range(n_eval):
            obs = env.reset()
            done = False
            total_reward = 0.0
            total_steps = 0

            while not done and total_steps < cfg.env.max_steps:
                obs_data = obs_parser.parse(obs)
                prompt = obs_to_prompt(obs_data)
                total_actions += 1

                if generator is not None:
                    json_blob = _run_constrained_generation(
                        generator, prompt, temperature, max_new_tokens,
                    )
                else:
                    json_blob = _run_unconstrained_generation(
                        model, tokenizer, prompt, device, max_new_tokens, temperature,
                    )

                if json_blob is None:
                    action = env.action_space({})
                    total_parse_fail += 1
                else:
                    try:
                        action = parse_tool_call(json_blob, env.action_space)
                    except Exception:
                        action = env.action_space({})
                        total_parse_fail += 1

                obs, reward, done, _ = env.step(action)
                total_reward += float(reward)
                total_steps += 1

            episode_metrics.append(
                {
                    "episode": ep,
                    "steps": total_steps,
                    "reward": total_reward,
                    "survived": (not done) and total_steps >= cfg.env.max_steps,
                }
            )
    finally:
        env.close()

    survival_rate = sum(int(m["survived"]) for m in episode_metrics) / max(1, len(episode_metrics))
    mean_steps = sum(m["steps"] for m in episode_metrics) / max(1, len(episode_metrics))
    mean_reward = sum(m["reward"] for m in episode_metrics) / max(1, len(episode_metrics))
    parse_fail_rate = total_parse_fail / max(1, total_actions)

    print(f"Eval over {n_eval} episodes:")
    print(f"  checkpoint    : {checkpoint_path}")
    print(f"  constrained   : {use_constrained}")
    print(f"  survival rate : {survival_rate:.2%}")
    print(f"  mean steps    : {mean_steps:.1f} / {cfg.env.max_steps}")
    print(f"  mean reward   : {mean_reward:.4f}")
    print(f"  parse failures: {total_parse_fail}/{total_actions} ({parse_fail_rate:.1%})")

    metrics_path = Path(cfg.output_dir) / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "episodes": episode_metrics,
                "constrained": use_constrained,
                "parse_fail_rate": parse_fail_rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  metrics file  : {metrics_path}")


if __name__ == "__main__":
    main()
