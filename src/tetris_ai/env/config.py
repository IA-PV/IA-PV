"""Immutable, serializable configuration for the Tetris environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from typing import Literal


CANONICAL_MAX_PIECES = 500
CANONICAL_RULESET_VERSION = "planning-v2"
CANONICAL_LINE_REWARDS = (0.0, 1.0, 3.0, 5.0, 8.0)

HorizonMode = Literal["finite", "time_limit"]


@dataclass(frozen=True)
class RewardConfig:
    """Reward terms kept separate from the game score."""

    line_rewards: tuple[float, float, float, float, float] = CANONICAL_LINE_REWARDS
    holes_delta_weight: float = 0.0
    aggregate_height_delta_weight: float = 0.0
    bumpiness_delta_weight: float = 0.0
    terminal_penalty: float = 0.0
    truncation_penalty: float = 0.0
    enable_shaping: bool = False
    shaping_gamma: float = 1.0
    shaping_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.line_rewards) != 5:
            raise ValueError("line_rewards must contain rewards for clearing 0 through 4 lines.")
        numeric_values = (
            *self.line_rewards,
            self.holes_delta_weight,
            self.aggregate_height_delta_weight,
            self.bumpiness_delta_weight,
            self.terminal_penalty,
            self.truncation_penalty,
            self.shaping_gamma,
            self.shaping_scale,
        )
        try:
            all_finite = all(math.isfinite(value) for value in numeric_values)
        except TypeError as error:
            raise ValueError("Reward values must be finite real numbers.") from error
        if not all_finite:
            raise ValueError("Reward values must be finite real numbers.")
        if not 0.0 <= self.shaping_gamma <= 1.0:
            raise ValueError("shaping_gamma must be between 0.0 and 1.0.")
        if self.shaping_scale < 0.0:
            raise ValueError("shaping_scale must be non-negative.")


def canonical_reward_config() -> RewardConfig:
    """Return the clean, stationary reward used by canonical evaluation."""

    return RewardConfig()


def rl_training_reward_config() -> RewardConfig:
    """Return the canonical task reward augmented with potential-based shaping."""

    return RewardConfig(
        holes_delta_weight=-0.50,
        aggregate_height_delta_weight=-0.05,
        bumpiness_delta_weight=-0.10,
        terminal_penalty=0.0,
        truncation_penalty=0.0,
        enable_shaping=True,
        shaping_gamma=1.0,
        shaping_scale=1.0,
    )


@dataclass(frozen=True)
class ScoringConfig:
    """Rules for the human-readable game score."""

    line_clear_scores: tuple[int, int, int, int, int] = (0, 100, 300, 500, 800)
    approximate_t_spin_scores: tuple[int, int, int, int, int] = (400, 800, 1200, 1600, 1600)
    back_to_back_multiplier: float = 1.5
    enable_approximate_t_spins: bool = False

    def __post_init__(self) -> None:
        if len(self.line_clear_scores) != 5 or len(self.approximate_t_spin_scores) != 5:
            raise ValueError("Scoring tables must contain values for clearing 0 through 4 lines.")
        if self.back_to_back_multiplier < 1.0:
            raise ValueError("back_to_back_multiplier must be at least 1.0.")


@dataclass(frozen=True)
class TetrisConfig:
    """Complete experiment contract for one environment instance."""

    width: int = 10
    height: int = 20
    max_pieces: int = CANONICAL_MAX_PIECES
    horizon_mode: HorizonMode = "finite"
    preview_count: int = 5
    allow_hold: bool = True
    observation_mode: Literal["raw", "featured"] = "featured"
    ruleset_version: str = CANONICAL_RULESET_VERSION
    reward: RewardConfig = field(default_factory=canonical_reward_config)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Board dimensions must be positive.")
        if self.max_pieces <= 0:
            raise ValueError("max_pieces must be positive.")
        if self.horizon_mode not in ("finite", "time_limit"):
            raise ValueError("horizon_mode must be 'finite' or 'time_limit'.")
        if self.preview_count <= 0:
            raise ValueError("preview_count must be positive.")
        if self.observation_mode not in ("raw", "featured"):
            raise ValueError("observation_mode must be 'raw' or 'featured'.")
        if not self.ruleset_version:
            raise ValueError("ruleset_version must not be empty.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]
