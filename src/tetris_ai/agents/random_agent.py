from __future__ import annotations

import random

from ..env.action import Action
from ..env.context import DecisionContext
from .base import Agent


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, context: DecisionContext) -> Action:
        actions = context.legal_actions
        if not actions:
            raise RuntimeError("No legal action is available.")
        return self._rng.choice(actions)
