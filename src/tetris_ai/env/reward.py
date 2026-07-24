"""Reward shaping, kept independent from game score."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..core.metrics import BoardMetrics
from .config import RewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable decomposition of task reward and optional PBRS terms.

    The legacy component names remain part of the public API even though each
    shaping value is now a potential difference rather than a plain metric
    delta.
    """

    line_reward: float
    holes_penalty: float
    aggregate_height_penalty: float
    bumpiness_penalty: float
    terminal_penalty: float
    truncation_penalty: float
    total: float
    task_reward: float = 0.0
    potential_shaping: float = 0.0

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
        holes_penalty = _potential_component(
            before.holes,
            after.holes,
            config.holes_delta_weight,
            terminated,
            config,
        )
        aggregate_height_penalty = _potential_component(
            before.aggregate_height,
            after.aggregate_height,
            config.aggregate_height_delta_weight,
            terminated,
            config,
        )
        bumpiness_penalty = _potential_component(
            before.bumpiness,
            after.bumpiness,
            config.bumpiness_delta_weight,
            terminated,
            config,
        )
    else:
        holes_penalty = aggregate_height_penalty = bumpiness_penalty = 0.0
    terminal_penalty = config.terminal_penalty if terminated else 0.0
    truncation_penalty = config.truncation_penalty if truncated else 0.0
    task_reward = line_reward + terminal_penalty + truncation_penalty
    potential_shaping = holes_penalty + aggregate_height_penalty + bumpiness_penalty
    total = task_reward + potential_shaping
    return RewardBreakdown(
        line_reward=line_reward,
        holes_penalty=holes_penalty,
        aggregate_height_penalty=aggregate_height_penalty,
        bumpiness_penalty=bumpiness_penalty,
        terminal_penalty=terminal_penalty,
        truncation_penalty=truncation_penalty,
        total=total,
        task_reward=task_reward,
        potential_shaping=potential_shaping,
    )


def _potential_component(
    before: int,
    after: int,
    weight: float,
    terminated: bool,
    config: RewardConfig,
) -> float:
    """Return ``scale * (gamma * Phi_i(s') - Phi_i(s))``.

    A true terminal state has zero potential. External truncations deliberately
    retain the successor potential so value-learning code can bootstrap from
    the observed boundary state.
    """

    successor_potential = 0.0 if terminated else config.shaping_gamma * weight * after
    current_potential = weight * before
    return config.shaping_scale * (successor_potential - current_potential)
