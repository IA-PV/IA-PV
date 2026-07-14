"""Reward shaping, kept independent from game score."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..core.metrics import BoardMetrics

LINE_REWARD = {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}


@dataclass(frozen=True)
class RewardBreakdown:
    line_reward: float
    holes_penalty: float
    aggregate_height_penalty: float
    bumpiness_penalty: float
    terminal_penalty: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_reward(before: BoardMetrics, after: BoardMetrics, lines_cleared: int, done: bool) -> RewardBreakdown:
    line_reward = LINE_REWARD[lines_cleared]
    holes_penalty = -0.75 * (after.holes - before.holes)
    aggregate_height_penalty = -0.10 * (after.aggregate_height - before.aggregate_height)
    bumpiness_penalty = -0.15 * (after.bumpiness - before.bumpiness)
    terminal_penalty = -10.0 if done else 0.0
    total = line_reward + holes_penalty + aggregate_height_penalty + bumpiness_penalty + terminal_penalty
    return RewardBreakdown(line_reward, holes_penalty, aggregate_height_penalty, bumpiness_penalty, terminal_penalty, total)
