"""Immutable, serializable configuration for the Tetris environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Literal


@dataclass(frozen=True)
class RewardConfig:
    """Reward terms kept separate from the game score."""

    line_rewards: tuple[float, float, float, float, float] = (0.0, 10.0, 30.0, 50.0, 80.0)
    holes_delta_weight: float = -0.75
    aggregate_height_delta_weight: float = -0.10
    bumpiness_delta_weight: float = -0.15
    terminal_penalty: float = -100.0
    truncation_penalty: float = 0.0
    enable_shaping: bool = True

    def __post_init__(self) -> None:
        if len(self.line_rewards) != 5:
            raise ValueError("line_rewards must contain rewards for clearing 0 through 4 lines.")


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
    max_pieces: int = 500
    preview_count: int = 5
    allow_hold: bool = True
    observation_mode: Literal["raw", "featured"] = "featured"
    ruleset_version: str = "planning-v1"
    reward: RewardConfig = field(default_factory=RewardConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Board dimensions must be positive.")
        if self.max_pieces <= 0:
            raise ValueError("max_pieces must be positive.")
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
