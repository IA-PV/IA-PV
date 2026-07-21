from __future__ import annotations

import random

from ..env.action import Action
from ..env.context import DecisionContext
from .base import Agent


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def configuration(self) -> dict[str, object]:
        return {"policy": "uniform_random_legal_action", "seed": self.seed}

    def select_action(self, context: DecisionContext) -> Action:
        actions = context.legal_actions
        if not actions:
            raise RuntimeError("No legal action is available.")
        return self._rng.choice(actions)
