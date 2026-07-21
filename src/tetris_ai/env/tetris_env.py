"""Turn-based, headless Tetris environment for planning and RL experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import random

from ..core.board import Board, Shape
from ..core.metrics import calculate_metrics
from ..core.randomizer import SevenBagRandomizer
from ..core.tetromino import PieceType, rotations_for
from .action import Action, EpisodeFinishedError, InvalidActionError
from .config import HorizonMode, TetrisConfig
from .context import DecisionContext, SimulatedTransition
from .observation import Observation
from .reward import calculate_reward
from .telemetry import ResetInfo, StepInfo


@dataclass(frozen=True)
class ActionPreview:
    """Public description used to animate or inspect a legal placement."""

    piece: PieceType
    shape: Shape
    row: int


@dataclass(frozen=True)
class _Placement:
    piece: PieceType
    shape: Shape
    row: int


class TetrisEnv:
    """Final-placement Tetris ruleset with a private seven-bag generator.

    One action always locks exactly one piece. When ``use_hold`` is true, the
    hold swap happens first and ``rotation``/``column`` place the replacement in
    the same transition. The ruleset intentionally has no gravity, hidden spawn
    rows, wall kicks, soft drops, or tucks.
    """

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        max_pieces: int | None = None,
        seed: int | None = None,
        *,
        preview_count: int | None = None,
        horizon_mode: HorizonMode | None = None,
        config: TetrisConfig | None = None,
    ) -> None:
        overrides = (width, height, max_pieces, preview_count, horizon_mode)
        if config is not None and any(value is not None for value in overrides):
            raise ValueError("Pass either config or individual environment overrides, not both.")
        if config is None:
            defaults = TetrisConfig()
            config = replace(
                defaults,
                width=defaults.width if width is None else width,
                height=defaults.height if height is None else height,
                max_pieces=defaults.max_pieces if max_pieces is None else max_pieces,
                preview_count=defaults.preview_count if preview_count is None else preview_count,
                horizon_mode=defaults.horizon_mode if horizon_mode is None else horizon_mode,
            )
        self.config = config
        self._randomizer: SevenBagRandomizer | None = None
        self._episode_seed: int | None = None
        self.reset(seed)

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def max_pieces(self) -> int:
        return self.config.max_pieces

    @property
    def horizon_mode(self) -> HorizonMode:
        return self.config.horizon_mode

    @property
    def remaining_pieces(self) -> int:
        return max(self.max_pieces - self.total_pieces_placed, 0)

    @property
    def action_space_size(self) -> int:
        return 8 * self.width

    @property
    def current_piece(self) -> PieceType | None:
        return self._current_piece

    @property
    def next_pieces(self) -> tuple[PieceType, ...]:
        return tuple(self._preview)

    @property
    def next_piece(self) -> PieceType | None:
        return self._preview[0] if self._preview else None

    @property
    def hold_piece(self) -> PieceType | None:
        return self._hold_piece

    @property
    def score(self) -> int:
        return self._score

    @property
    def total_lines_cleared(self) -> int:
        return self._total_lines_cleared

    @property
    def total_pieces_placed(self) -> int:
        return self._total_pieces_placed

    @property
    def back_to_back_active(self) -> bool:
        return self._back_to_back_active

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def termination_reasons(self) -> tuple[str, ...]:
        return self._terminal_reasons + self._truncation_reasons

    @property
    def termination_reason(self) -> str | None:
        return "+".join(self.termination_reasons) or None

    @property
    def level(self) -> int:
        return self.total_lines_cleared // 10 + 1

    @property
    def board_matrix(self) -> tuple[tuple[int, ...], ...]:
        return self._board.matrix()

    def reset(self, seed: int | None = None) -> tuple[Observation, ResetInfo]:
        reseeded = self._randomizer is None or seed is not None
        if reseeded:
            effective_seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
            self._randomizer = SevenBagRandomizer(effective_seed)
            self._episode_seed = effective_seed
        assert self._randomizer is not None

        self._board = Board(self.width, self.height)
        self._current_piece: PieceType | None = self._randomizer.next_piece()
        self._preview = deque(self._randomizer.next_piece() for _ in range(self.config.preview_count))
        self._hold_piece: PieceType | None = None
        self._back_to_back_active = False
        self._score = 0
        self._total_lines_cleared = 0
        self._total_pieces_placed = 0
        self._terminated = False
        self._truncated = False
        self._terminal_reasons: tuple[str, ...] = ()
        self._truncation_reasons: tuple[str, ...] = ()
        self._last_metrics = calculate_metrics(self._board)
        self._legal_placement_cache: dict[Action, _Placement] | None = None
        if not self._legal_placements_after_state_update():
            self._terminal_reasons = ("game_over",)
            self._terminated = True

        observation = self.get_observation()
        info: ResetInfo = {
            "seed": self._episode_seed,
            "rng_reseeded": reseeded,
            "ruleset_version": self.config.ruleset_version,
            "config_id": self.config.fingerprint,
            "config": self.config.as_dict(),
            "horizon_mode": self.horizon_mode,
            "max_pieces": self.max_pieces,
            "remaining_pieces": self.remaining_pieces,
            "state_hash": self.state_hash(),
        }
        return observation, info

    def _placements_for_piece(self, piece: PieceType, use_hold: bool) -> dict[Action, _Placement]:
        placements: dict[Action, _Placement] = {}
        for rotation, shape in enumerate(rotations_for(piece)):
            shape_width = len(shape[0])
            for column in range(self.width - shape_width + 1):
                row = self._board.drop_row(shape, column)
                if row is not None:
                    action = Action(rotation=rotation, column=column, use_hold=use_hold)
                    placements[action] = _Placement(piece=piece, shape=shape, row=row)
        return placements

    def _legal_placements_after_state_update(self) -> dict[Action, _Placement]:
        if self._legal_placement_cache is not None:
            return self._legal_placement_cache
        placements: dict[Action, _Placement] = {}
        if self.current_piece is not None:
            placements.update(self._placements_for_piece(self.current_piece, use_hold=False))
            replacement = self.hold_piece if self.hold_piece is not None else self.next_piece
            if self.config.allow_hold and replacement is not None:
                placements.update(self._placements_for_piece(replacement, use_hold=True))
        self._legal_placement_cache = placements
        return placements

    def legal_actions(self) -> tuple[Action, ...]:
        if self.done and not self._exposes_time_limit_bootstrap_state():
            return ()
        return tuple(self._legal_placements_after_state_update())

    def _exposes_time_limit_bootstrap_state(self) -> bool:
        return (
            not self.terminated
            and self.truncated
            and self.horizon_mode == "time_limit"
            and self._truncation_reasons == ("piece_limit",)
        )

    def action_mask(self) -> tuple[bool, ...]:
        mask = [False] * self.action_space_size
        for action in self.legal_actions():
            mask[action.to_id(self.width)] = True
        return tuple(mask)

    def describe_action(self, action: Action) -> ActionPreview:
        if action not in self.legal_actions():
            raise InvalidActionError(f"Action {action!r} is not legal in the current state.")
        placement = self._legal_placements_after_state_update()[action]
        return ActionPreview(placement.piece, placement.shape, placement.row)

    def get_observation(self) -> Observation:
        return Observation(
            board=self.board_matrix,
            current_piece=self.current_piece,
            next_pieces=self.next_pieces,
            hold_piece=self.hold_piece,
            can_hold=any(action.use_hold for action in self.legal_actions()),
            score=self.score,
            level=self.level,
            total_lines_cleared=self.total_lines_cleared,
            total_pieces_placed=self.total_pieces_placed,
            max_pieces=self.max_pieces,
            remaining_pieces=self.remaining_pieces,
            back_to_back_active=self.back_to_back_active,
            terminated=self.terminated,
            truncated=self.truncated,
            action_mask=self.action_mask(),
            metrics=self._last_metrics if self.config.observation_mode == "featured" else None,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, StepInfo]:
        if self.done:
            raise EpisodeFinishedError("Cannot step a finished Tetris episode; call reset().")
        placements = self._legal_placements_after_state_update()
        placement = placements.get(action)
        if placement is None:
            piece_name = self.current_piece.value if self.current_piece is not None else "none"
            raise InvalidActionError(f"Action {action!r} is not legal for {piece_name}.")

        before = self._last_metrics
        level_before = self.level
        legal_action_count = len(placements)
        previous_current = self.current_piece
        assert previous_current is not None

        if action.use_hold:
            if self.hold_piece is None:
                self._hold_piece = previous_current
                placed_piece = self._take_preview_piece()
            else:
                placed_piece = self.hold_piece
                self._hold_piece = previous_current
        else:
            placed_piece = previous_current
        assert placed_piece == placement.piece

        self._board.place(placement.shape, action.column, placement.row)
        approximate_t_spin = self.config.scoring.enable_approximate_t_spins and self._is_approximate_t_spin(
            placed_piece, placement.shape, action.column, placement.row
        )
        lines_cleared = self._board.clear_lines()
        score_gain, b2b_bonus_applied = self._score_for_lock(
            lines_cleared, approximate_t_spin, level_before
        )
        self._score += score_gain
        self._total_lines_cleared += lines_cleared
        self._total_pieces_placed += 1
        self._current_piece = self._take_preview_piece()

        self._invalidate_legal_actions()
        has_legal_placements = self.current_piece is not None and bool(
            self._legal_placements_after_state_update()
        )
        terminal_reasons: list[str] = []
        if self.current_piece is not None and not has_legal_placements:
            terminal_reasons.append("game_over")
        if self.total_pieces_placed >= self.max_pieces and self.horizon_mode == "finite":
            terminal_reasons.append("horizon_completed")
        self._terminal_reasons = tuple(terminal_reasons)
        self._terminated = bool(terminal_reasons)

        truncation_reasons: list[str] = []
        if self.total_pieces_placed >= self.max_pieces and self.horizon_mode == "time_limit":
            truncation_reasons.append("piece_limit")
        if self.current_piece is None:
            truncation_reasons.append("preview_horizon")
        self._truncation_reasons = tuple(truncation_reasons)
        self._truncated = bool(truncation_reasons)

        self._last_metrics = calculate_metrics(self._board, lines_cleared)
        breakdown = calculate_reward(
            before,
            self._last_metrics,
            lines_cleared,
            self.terminated,
            self.truncated,
            self.config.reward,
        )
        info: StepInfo = {
            "action": action.as_dict(self.width),
            "placed_piece": placed_piece.value,
            "placed_row": placement.row,
            "lines_cleared": lines_cleared,
            "score_gain": score_gain,
            "level_before": level_before,
            "level_after": self.level,
            "is_t_spin": approximate_t_spin,
            "t_spin_detection": "three_corner_approximation" if approximate_t_spin else None,
            "back_to_back_active": self.back_to_back_active,
            "back_to_back_bonus_applied": b2b_bonus_applied,
            "hold_used": action.use_hold,
            "held_piece": self.hold_piece.value if self.hold_piece is not None else None,
            "board_metrics_before": before.as_dict(),
            "board_metrics_after": self._last_metrics.as_dict(),
            "legal_action_count": legal_action_count,
            "reward": breakdown.as_dict(),
            "horizon_mode": self.horizon_mode,
            "max_pieces": self.max_pieces,
            "remaining_pieces": self.remaining_pieces,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "termination_reasons": self.termination_reasons,
            "state_hash": self.state_hash(),
            "config_id": self.config.fingerprint,
        }
        return self.get_observation(), breakdown.total, self.terminated, self.truncated, info

    def _take_preview_piece(self) -> PieceType | None:
        if not self._preview:
            return None
        piece = self._preview.popleft()
        if self._randomizer is not None:
            self._preview.append(self._randomizer.next_piece())
        return piece

    def _invalidate_legal_actions(self) -> None:
        self._legal_placement_cache = None

    def _score_for_lock(self, lines_cleared: int, is_t_spin: bool, level: int | None = None) -> tuple[int, bool]:
        scoring = self.config.scoring
        score_level = self.level if level is None else level
        base_score = (
            scoring.approximate_t_spin_scores[lines_cleared]
            if is_t_spin
            else scoring.line_clear_scores[lines_cleared]
        )
        b2b_eligible = lines_cleared > 0 and (is_t_spin or lines_cleared == 4)
        b2b_bonus_applied = b2b_eligible and self._back_to_back_active
        multiplier = scoring.back_to_back_multiplier if b2b_bonus_applied else 1.0
        if b2b_eligible:
            self._back_to_back_active = True
        elif lines_cleared > 0:
            self._back_to_back_active = False
        return int(base_score * score_level * multiplier), b2b_bonus_applied

    def _is_approximate_t_spin(
        self,
        piece: PieceType,
        shape: Shape,
        column: int,
        row: int,
    ) -> bool:
        if piece != PieceType.T:
            return False
        center = self._t_center(shape)
        if center is None:
            return False
        center_row, center_column = center
        board_row = row + center_row
        board_column = column + center_column
        corners = (
            (board_row - 1, board_column - 1),
            (board_row - 1, board_column + 1),
            (board_row + 1, board_column - 1),
            (board_row + 1, board_column + 1),
        )
        return sum(self._board.is_filled_or_wall(r, c) for r, c in corners) >= 3

    @staticmethod
    def _t_center(shape: Shape) -> tuple[int, int] | None:
        for row, cells in enumerate(shape):
            for column, occupied in enumerate(cells):
                if not occupied:
                    continue
                neighbors = (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                )
                occupied_neighbors = sum(
                    1
                    for neighbor_row, neighbor_column in neighbors
                    if TetrisEnv._shape_cell(shape, neighbor_row, neighbor_column)
                )
                if occupied_neighbors >= 3:
                    return row, column
        return None

    @staticmethod
    def _shape_cell(shape: Shape, row: int, column: int) -> bool:
        return 0 <= row < len(shape) and 0 <= column < len(shape[row]) and bool(shape[row][column])

    def state_hash(self) -> str:
        payload = {
            "board": self.board_matrix,
            "current_piece": self.current_piece.value if self.current_piece is not None else None,
            "next_pieces": [piece.value for piece in self.next_pieces],
            "hold_piece": self.hold_piece.value if self.hold_piece is not None else None,
            "score": self.score,
            "lines": self.total_lines_cleared,
            "pieces": self.total_pieces_placed,
            "back_to_back": self.back_to_back_active,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reasons": self.termination_reasons,
            "config_id": self.config.fingerprint,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def decision_context(self) -> DecisionContext:
        """Return an agent-safe model containing only the public preview queue."""

        planning_model = self.clone()
        planning_model._randomizer = None
        return planning_model._decision_context_from_model()

    def _decision_context_from_model(self) -> DecisionContext:
        return DecisionContext(
            observation=self.get_observation(),
            legal_actions=self.legal_actions(),
            _simulator=self._simulate_public_action,
        )

    def _simulate_public_action(self, action: Action) -> SimulatedTransition:
        simulation = self.clone()
        _, reward, terminated, truncated, info = simulation.step(action)
        return SimulatedTransition(
            context=simulation._decision_context_from_model(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def clone(self) -> "TetrisEnv":
        """Create an exact administrative clone, including RNG state when present."""

        clone = TetrisEnv.__new__(TetrisEnv)
        clone.config = self.config
        clone._randomizer = self._randomizer.clone() if self._randomizer is not None else None
        clone._episode_seed = self._episode_seed
        clone._board = self._board.clone()
        clone._current_piece = self.current_piece
        clone._preview = deque(self._preview)
        clone._hold_piece = self.hold_piece
        clone._back_to_back_active = self.back_to_back_active
        clone._score = self.score
        clone._total_lines_cleared = self.total_lines_cleared
        clone._total_pieces_placed = self.total_pieces_placed
        clone._terminated = self.terminated
        clone._truncated = self.truncated
        clone._terminal_reasons = self._terminal_reasons
        clone._truncation_reasons = self._truncation_reasons
        clone._last_metrics = self._last_metrics
        clone._legal_placement_cache = None
        return clone
