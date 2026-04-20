"""Rollout collection: sample G tool calls per state, step the environment."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

import torch

from gridzero.env.observation import ObsData
from gridzero.training.buffer import RolloutBuffer, StoredTransition


@dataclass
class Transition:
    """One timestep: G completions sampled for a single grid state."""

    obs: ObsData
    embeddings: torch.Tensor            # [N, d_model] from encoder
    completions: list[str]              # G JSON tool call strings
    old_log_probs: list[float]          # G scalar log-probs under rollout policy
    rewards: list[float]                # G scalar rewards
    dones: list[bool]                   # whether each completion ended the episode


@dataclass
class Episode:
    transitions: list[Transition] = field(default_factory=list)
    survived: bool = False
    total_steps: int = 0


class RolloutCollector:
    """Collects episodes by running the policy against the environment.

    For each timestep, samples G completions from vllm, steps G copies of the
    environment (one per completion), and stores all transitions.

    The single-best (highest reward) action is used to advance the main episode;
    all G (obs, completion, reward) tuples go into the training buffer.
    """

    def __init__(self, encoder, vllm_client, json_schema: dict, cfg) -> None:
        self.encoder = encoder
        self.vllm_client = vllm_client
        self.json_schema = json_schema
        self.cfg = cfg

    async def collect_episode(
        self,
        env,
        action_space,
        prompt: str,
        buffer: RolloutBuffer,
        group_id_start: int = 0,
    ) -> Episode:
        """Run one full episode and populate the rollout buffer.

        Args:
            env: grid2op GymEnv (reset before calling).
            action_space: grid2op action_space for parsing tool calls.
            prompt: Formatted prompt string from tokenizer_utils.
            buffer: Buffer to append StoredTransitions into.
            group_id_start: Starting group ID for this episode's transitions.

        Returns:
            Episode summary.
        """
        from gridzero.env.actions import parse_tool_call
        from gridzero.policy.logprob import sequence_log_prob

        obs_raw, _ = env.reset()
        episode = Episode()
        group_id = group_id_start
        done = False

        while not done and episode.total_steps < self.cfg.env.max_steps:
            # Encode current observation
            embeddings = self.encoder(obs_raw)

            # Sample G completions from vllm
            completions = await self.vllm_client.sample_completions(
                embeddings=embeddings,
                prompt=prompt,
                n=self.cfg.training.n_completions_per_state,
                json_schema=self.json_schema,
            )

            # Evaluate each completion: step a copy of the env for each
            rewards: list[float] = []
            dones: list[bool] = []
            for completion in completions:
                try:
                    action = parse_tool_call(completion, action_space)
                except Exception:
                    # Invalid action (out-of-range IDs etc) — treat as do_nothing
                    action = action_space({})
                _, r, d, _, _ = env.step(action)
                # TODO: use env copies so the main episode is not consumed here
                rewards.append(float(r))
                dones.append(bool(d))

            # TODO: compute old_log_probs via reference policy forward pass
            old_log_probs = [0.0] * len(completions)

            # Advance main episode with best action (greedy at rollout time)
            best_idx = int(max(range(len(rewards)), key=lambda i: rewards[i]))
            best_action = parse_tool_call(completions[best_idx], action_space)
            obs_raw, _, done, _, _ = env.step(best_action)

            # Add all G completions to the buffer
            for i, (comp, lp, r) in enumerate(zip(completions, old_log_probs, rewards)):
                buffer.add(StoredTransition(
                    prompt_ids=torch.tensor([]),   # TODO: tokenize prompt
                    completion_ids=torch.tensor([]),  # TODO: tokenize completion
                    old_log_prob=lp,
                    reward=r,
                    group_id=group_id,
                ))

            episode.transitions.append(
                Transition(
                    obs=obs_raw,
                    embeddings=embeddings,
                    completions=completions,
                    old_log_probs=old_log_probs,
                    rewards=rewards,
                    dones=dones,
                )
            )
            episode.total_steps += 1
            group_id += 1

        episode.survived = not done
        return episode

    async def collect_batch(
        self,
        env_factory: Callable,
        action_space,
        prompt: str,
        buffer: RolloutBuffer,
        n_episodes: int,
    ) -> list[Episode]:
        """Collect n_episodes concurrently (one coroutine per episode)."""
        tasks = [
            self.collect_episode(
                env=env_factory(),
                action_space=action_space,
                prompt=prompt,
                buffer=buffer,
                group_id_start=i * self.cfg.env.max_steps,
            )
            for i in range(n_episodes)
        ]
        return await asyncio.gather(*tasks)
