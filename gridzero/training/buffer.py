"""Rollout buffer for storing and batching GSPO training data."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class StoredTransition:
    """Flattened, tensor-ready version of a Transition for batch processing."""
    prompt_ids: torch.Tensor        # [T_prompt]
    completion_ids: torch.Tensor    # [T_comp]
    old_log_prob: float             # scalar, log P under rollout policy
    reward: float                   # scalar composite reward
    group_id: int                   # which state/group this came from


class RolloutBuffer:
    """Stores transitions from multiple episodes for one GSPO update step.

    Transitions are grouped by state (each state contributes G completions).
    The buffer is cleared after each optimizer step.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._transitions: list[StoredTransition] = []

    def add(self, transition: StoredTransition) -> None:
        """Add a single transition to the buffer."""
        if len(self._transitions) < self.max_size:
            self._transitions.append(transition)

    def clear(self) -> None:
        """Empty the buffer after an update step."""
        self._transitions.clear()

    def as_tensors(self) -> dict[str, torch.Tensor]:
        """Stack all stored transitions into batched tensors.

        Returns dict with keys:
            old_log_probs: [total_transitions]
            rewards:       [total_transitions]
            group_ids:     [total_transitions]
        Plus jagged lists for prompt_ids and completion_ids (variable length).
        """
        old_log_probs = torch.tensor([t.old_log_prob for t in self._transitions])
        rewards = torch.tensor([t.reward for t in self._transitions])
        group_ids = torch.tensor([t.group_id for t in self._transitions])
        prompt_ids = [t.prompt_ids for t in self._transitions]
        completion_ids = [t.completion_ids for t in self._transitions]
        return {
            "old_log_probs": old_log_probs,
            "rewards": rewards,
            "group_ids": group_ids,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        }

    def __len__(self) -> int:
        return len(self._transitions)

    def is_full(self) -> bool:
        return len(self._transitions) >= self.max_size
