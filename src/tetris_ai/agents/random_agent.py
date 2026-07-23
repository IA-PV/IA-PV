"""Seeded random-policy baseline for Tetris evaluations."""

from __future__ import annotations

import random

from ..env.action import Action
from ..env.context import DecisionContext
from .base import Agent


class RandomAgent(Agent):
    """Choose uniformly among the legal actions using an isolated PRNG.

    The baseline is deterministic for a given seed and never consumes the
    environment's random state, which preserves paired-seed comparisons.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def configuration(self) -> dict[str, object]:
        return {"algorithm": "uniform_random_legal_action", "seed": self.seed}

    def select_action(self, context: DecisionContext) -> Action:
        if not context.legal_actions:
            raise ValueError("RandomAgent requires at least one legal action.")
        return self._rng.choice(context.legal_actions)
