"""Simple rollout buffer used for evaluation and smoke checks."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Transition:
    reward: float
    done: bool


class RolloutBuffer:
    """In-memory transition store with tensor export helpers."""

    def __init__(self, max_size: int = 100_000) -> None:
        self.max_size = max_size
        self._items: list[Transition] = []

    def add(self, reward: float, done: bool) -> None:
        if len(self._items) >= self.max_size:
            self._items.pop(0)
        self._items.append(Transition(reward=float(reward), done=bool(done)))

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def as_tensors(self) -> dict[str, torch.Tensor]:
        if not self._items:
            return {
                "rewards": torch.zeros(0, dtype=torch.float32),
                "dones": torch.zeros(0, dtype=torch.bool),
            }
        rewards = torch.tensor([t.reward for t in self._items], dtype=torch.float32)
        dones = torch.tensor([t.done for t in self._items], dtype=torch.bool)
        return {"rewards": rewards, "dones": dones}
