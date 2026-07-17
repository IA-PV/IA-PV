"""Reward shaping, kept independent from game score."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..core.metrics import BoardMetrics
from .config import RewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    line_reward: float
    holes_penalty: float
    aggregate_height_penalty: float
    bumpiness_penalty: float
    terminal_penalty: float
    truncation_penalty: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_reward(
    before: BoardMetrics,
    after: BoardMetrics,
    lines_cleared: int,
    terminated: bool,
    truncated: bool,
    config: RewardConfig,
) -> RewardBreakdown:
    line_reward = config.line_rewards[lines_cleared]
    if config.enable_shaping:
        holes_penalty = config.holes_delta_weight * (after.holes - before.holes)
        aggregate_height_penalty = config.aggregate_height_delta_weight * (
            after.aggregate_height - before.aggregate_height
        )
        bumpiness_penalty = config.bumpiness_delta_weight * (after.bumpiness - before.bumpiness)
    else:
        holes_penalty = aggregate_height_penalty = bumpiness_penalty = 0.0
    terminal_penalty = config.terminal_penalty if terminated else 0.0
    truncation_penalty = config.truncation_penalty if truncated else 0.0
    total = (
        line_reward
        + holes_penalty
        + aggregate_height_penalty
        + bumpiness_penalty
        + terminal_penalty
        + truncation_penalty
    )
    return RewardBreakdown(
        line_reward,
        holes_penalty,
        aggregate_height_penalty,
        bumpiness_penalty,
        terminal_penalty,
        truncation_penalty,
        total,
    )
