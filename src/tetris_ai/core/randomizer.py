"""Deterministic seven-bag tetromino randomizer."""

from __future__ import annotations

import random

from .tetromino import PieceType


class SevenBagRandomizer:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._bag: list[PieceType] = []

    def next_piece(self) -> PieceType:
        if not self._bag:
            self._bag = list(PieceType)
            self._rng.shuffle(self._bag)
        return self._bag.pop()

    def clone(self) -> "SevenBagRandomizer":
        clone = SevenBagRandomizer()
        clone._rng.setstate(self._rng.getstate())
        clone._bag = self._bag.copy()
        return clone
