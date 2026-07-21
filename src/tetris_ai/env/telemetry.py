"""Typed, JSON-compatible reset and transition telemetry."""

from __future__ import annotations

from typing import TypedDict

from .config import HorizonMode


class ResetInfo(TypedDict):
    seed: int | None
    rng_reseeded: bool
    ruleset_version: str
    config_id: str
    config: dict[str, object]
    horizon_mode: HorizonMode
    max_pieces: int
    remaining_pieces: int
    state_hash: str


class StepInfo(TypedDict):
    action: dict[str, int | bool]
    placed_piece: str
    placed_row: int
    lines_cleared: int
    score_gain: int
    level_before: int
    level_after: int
    is_t_spin: bool
    t_spin_detection: str | None
    back_to_back_active: bool
    back_to_back_bonus_applied: bool
    hold_used: bool
    held_piece: str | None
    board_metrics_before: dict[str, int | tuple[int, ...]]
    board_metrics_after: dict[str, int | tuple[int, ...]]
    legal_action_count: int
    reward: dict[str, float]
    horizon_mode: HorizonMode
    max_pieces: int
    remaining_pieces: int
    terminated: bool
    truncated: bool
    termination_reason: str | None
    termination_reasons: tuple[str, ...]
    state_hash: str
    config_id: str
