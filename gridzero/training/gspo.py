"""Group Sequence Policy Optimization (GSPO) loss and training loop."""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from gridzero.training.buffer import RolloutBuffer


def gspo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GSPO sequence-level clipped policy gradient loss.

    Each group (identified by group_ids) corresponds to one grid state with G
    completions. Rewards are normalized within each group to form advantages.
    Importance ratios are computed at the full-sequence level (sum of token
    log-probs), not per-token.

    Args:
        log_probs:     Current policy log P(completion|state), shape [B*G].
        old_log_probs: Reference (rollout) policy log probs, shape [B*G].
        rewards:       Scalar reward per completion, shape [B*G].
        group_ids:     Integer group index per completion, shape [B*G].
        clip_eps:      PPO-style importance ratio clipping epsilon.
        kl_coeff:      KL divergence penalty coefficient.

    Returns:
        (loss scalar, metrics dict)
    """
    # Group-normalize rewards into advantages
    advantages = _group_normalize(rewards, group_ids)

    # Sequence-level importance ratios
    ratio = torch.exp(log_probs - old_log_probs.detach())

    # Clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)

    # Approximate KL penalty: E[ratio - 1 - log(ratio)]
    kl = ratio - 1.0 - torch.log(ratio + 1e-8)

    loss = -(surrogate - kl_coeff * kl).mean()

    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
    metrics = {
        "loss": loss.item(),
        "mean_advantage": advantages.mean().item(),
        "mean_ratio": ratio.mean().item(),
        "clip_fraction": clip_frac,
        "mean_kl": kl.mean().item(),
    }
    return loss, metrics


def _group_normalize(rewards: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """Normalize rewards within each group to zero mean, unit std (advantages)."""
    advantages = torch.zeros_like(rewards)
    for gid in group_ids.unique():
        mask = group_ids == gid
        group_r = rewards[mask]
        std = group_r.std().clamp(min=1e-8)
        advantages[mask] = (group_r - group_r.mean()) / std
    return advantages


class GSPOTrainer:
    """Manages the full GSPO training loop.

    Responsibilities:
    - Maintaining current policy and frozen reference policy
    - Running rollout collection (via RolloutCollector)
    - Computing GSPO loss and stepping the optimizer
    - Logging metrics
    """

    def __init__(
        self,
        policy,
        encoder,
        vllm_client,
        env_factory,
        cfg: DictConfig,
    ) -> None:
        self.policy = policy
        self.encoder = encoder
        self.vllm_client = vllm_client
        self.env_factory = env_factory
        self.cfg = cfg

        self.optimizer = torch.optim.AdamW(
            list(policy.model.parameters()) + list(encoder.parameters()),
            lr=cfg.training.lr,
        )

        # Frozen reference policy for KL computation
        self.ref_policy = copy.deepcopy(policy.model)
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

    def train_step(self, buffer: RolloutBuffer) -> dict[str, float]:
        """One GSPO parameter update from a filled rollout buffer.

        Returns:
            Dict of training metrics.
        """
        from gridzero.policy.logprob import sequence_log_prob

        batch = buffer.as_tensors()

        # Re-compute log-probs under current policy
        # TODO: batch tokenized completions and run sequence_log_prob
        log_probs = torch.zeros(len(buffer))  # placeholder

        loss, metrics = gspo_loss(
            log_probs=log_probs,
            old_log_probs=batch["old_log_probs"],
            rewards=batch["rewards"],
            group_ids=batch["group_ids"],
            clip_eps=self.cfg.training.clip_eps,
            kl_coeff=self.cfg.training.kl_coeff,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(), self.cfg.training.grad_clip
        )
        self.optimizer.step()

        return metrics

    def run(self, n_updates: int) -> None:
        """Full training loop: collect → update → repeat.

        Args:
            n_updates: Number of optimizer steps to run.
        """
        import asyncio

        from gridzero.inference.constrained_gen import build_guided_json_params
        from gridzero.policy.tokenizer_utils import build_prompt
        from gridzero.training.rollout import RolloutCollector

        prompt = build_prompt()
        json_schema = build_guided_json_params()
        collector = RolloutCollector(
            encoder=self.encoder,
            vllm_client=self.vllm_client,
            json_schema=json_schema,
            cfg=self.cfg,
        )
        buffer = RolloutBuffer(max_size=self.cfg.training.episodes_per_update * self.cfg.env.max_steps * self.cfg.training.n_completions_per_state)

        for update in range(n_updates):
            buffer.clear()
            episodes = asyncio.run(
                collector.collect_batch(
                    env_factory=self.env_factory,
                    action_space=self.env_factory().action_space,
                    prompt=prompt,
                    buffer=buffer,
                    n_episodes=self.cfg.training.episodes_per_update,
                )
            )
            metrics = self.train_step(buffer)
            survived = sum(e.survived for e in episodes)
            print(
                f"[update {update+1}/{n_updates}] "
                f"loss={metrics['loss']:.4f} "
                f"mean_reward={buffer.as_tensors()['rewards'].mean():.4f} "
                f"survived={survived}/{len(episodes)} "
                f"clip_frac={metrics['clip_fraction']:.3f}"
            )
