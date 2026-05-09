"""Async rollout collection utilities for evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gridzero.env.actions import parse_tool_call
from gridzero.rewards.grid_rewards import composite_reward


@dataclass
class EpisodeStats:
    total_steps: int
    total_reward: float
    survived: bool


class RolloutCollector:
    """Collect episodes from a grid2op env using encoder + vllm completions."""

    def __init__(self, encoder, vllm_client, json_schema: dict, cfg) -> None:
        self.encoder = encoder
        self.vllm_client = vllm_client
        self.json_schema = json_schema
        self.cfg = cfg

    async def collect_batch(
        self,
        env_factory: Callable,
        action_space,
        prompt: str,
        buffer,
        n_episodes: int,
    ) -> list[EpisodeStats]:
        episodes: list[EpisodeStats] = []

        for _ in range(n_episodes):
            env, obs_parser = env_factory()
            obs = env.reset()
            done = False
            total_reward = 0.0
            total_steps = 0

            while not done and total_steps < self.cfg.env.max_steps:
                obs_data = obs_parser.parse(obs)
                try:
                    embeddings = self.encoder(obs_data)
                    completions = await self.vllm_client.sample_completions(
                        embeddings=embeddings,
                        prompt=prompt,
                        n=1,
                        json_schema=self.json_schema,
                        max_tokens=self.cfg.training.get("max_generation_tokens", 128),
                        temperature=self.cfg.training.get("temperature", 0.7),
                    )
                    completion = completions[0]
                    action = parse_tool_call(completion, action_space)
                except Exception:
                    # Fallback to do-nothing for smoke robustness.
                    action = action_space({})

                obs, env_reward, done, _ = env.step(action)
                shaped_reward = composite_reward(obs, done, max_steps=self.cfg.env.max_steps)
                total_reward += float(shaped_reward)
                total_steps += 1
                buffer.add(reward=float(shaped_reward), done=done)

            survived = not done and total_steps >= self.cfg.env.max_steps
            episodes.append(
                EpisodeStats(
                    total_steps=total_steps,
                    total_reward=total_reward,
                    survived=survived,
                )
            )
            env.close()

        return episodes
