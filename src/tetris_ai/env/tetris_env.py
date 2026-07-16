"""Turn-based, headless Tetris environment."""

from __future__ import annotations

from ..core.board import Board
from ..core.metrics import BoardMetrics, calculate_metrics
from ..core.randomizer import SevenBagRandomizer
from ..core.tetromino import PieceType, rotations_for
from .action import Action, EpisodeFinishedError, InvalidActionError
from .observation import Observation
from .reward import RewardBreakdown, calculate_reward

LINE_CLEAR_SCORE = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
T_SPIN_SCORE = {0: 400, 1: 800, 2: 1200, 3: 1600, 4: 1600}
B2B_MULTIPLIER = 1.5


class TetrisEnv:
    def __init__(self, width: int = 10, height: int = 20, max_pieces: int = 500, seed: int | None = None) -> None:
        if max_pieces <= 0:
            raise ValueError("max_pieces must be positive.")
        self.width, self.height, self.max_pieces = width, height, max_pieces
        self._initial_seed = seed
        self.reset(seed)

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._initial_seed = seed
        self.board = Board(self.width, self.height)
        self._randomizer = SevenBagRandomizer(self._initial_seed)
        self.current_piece = self._randomizer.next_piece()
        self.next_piece = self._randomizer.next_piece()
        self.hold_piece: PieceType | None = None
        self.can_hold = True
        self.back_to_back_active = False
        self.score = 0
        self.total_lines_cleared = 0
        self.total_pieces_placed = 0
        self.done = not self._legal_actions_after_state_update()
        self.termination_reason: str | None = "game_over" if self.done else None
        self._last_metrics = calculate_metrics(self.board)
        return self.get_observation()

    @property
    def level(self) -> int:
        return self.total_lines_cleared // 10 + 1

    def _actions_for_piece(self, piece: PieceType) -> tuple[Action, ...]:
        actions: list[Action] = []
        for rotation, shape in enumerate(rotations_for(piece)):
            shape_width = len(shape[0])
            for column in range(self.width - shape_width + 1):
                if self.board.drop_row(shape, column) is not None:
                    actions.append(Action(rotation, column))
        return tuple(actions)

    def _actions_for_current_piece(self) -> tuple[Action, ...]:
        return self._actions_for_piece(self.current_piece)

    def legal_actions(self) -> tuple[Action, ...]:
        if self.done:
            return ()
        return self._legal_actions_after_state_update()

    def _legal_actions_after_state_update(self) -> tuple[Action, ...]:
        actions = list(self._actions_for_current_piece())
        if self._can_legally_hold():
            actions.append(Action.hold())
        return tuple(actions)

    def _can_legally_hold(self) -> bool:
        if not self.can_hold:
            return False
        held_replacement = self.hold_piece if self.hold_piece is not None else self.next_piece
        return bool(self._actions_for_piece(held_replacement))

    def get_observation(self) -> Observation:
        return Observation(
            board=self.board.matrix(),
            current_piece=self.current_piece,
            next_piece=self.next_piece,
            hold_piece=self.hold_piece,
            can_hold=self.can_hold,
            score=self.score,
            level=self.level,
            total_lines_cleared=self.total_lines_cleared,
            total_pieces_placed=self.total_pieces_placed,
            back_to_back_active=self.back_to_back_active,
            done=self.done,
            metrics=self._last_metrics,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self.done:
            raise EpisodeFinishedError("Cannot step a finished Tetris episode; call reset().")
        if action not in self.legal_actions():
            raise InvalidActionError(f"Action {action!r} is not legal for {self.current_piece.value}.")
        if action.is_hold:
            return self._step_hold(action)
        before = calculate_metrics(self.board)
        placed_piece = self.current_piece
        shape = rotations_for(self.current_piece)[action.rotation]
        row = self.board.drop_row(shape, action.column)
        assert row is not None  # guaranteed by legal_actions
        self.board.place(shape, action.column, row)
        is_t_spin = self._is_t_spin(placed_piece, shape, action.column, row)
        lines_cleared = self.board.clear_lines()
        self.total_lines_cleared += lines_cleared
        score_gain, b2b_bonus_applied = self._score_for_lock(lines_cleared, is_t_spin)
        self.score += score_gain
        self.total_pieces_placed += 1
        self.can_hold = True

        self.current_piece = self.next_piece
        self.next_piece = self._randomizer.next_piece()
        if self.total_pieces_placed >= self.max_pieces:
            self.done, self.termination_reason = True, "piece_limit"
        elif not self._legal_actions_after_state_update():
            self.done, self.termination_reason = True, "game_over"
        else:
            self.done, self.termination_reason = False, None

        self._last_metrics = calculate_metrics(self.board, lines_cleared)
        breakdown = calculate_reward(before, self._last_metrics, lines_cleared, self.done)
        info = {
            "action": action,
            "lines_cleared": lines_cleared,
            "score_gain": score_gain,
            "level": self.level,
            "is_t_spin": is_t_spin,
            "back_to_back_active": self.back_to_back_active,
            "back_to_back_bonus_applied": b2b_bonus_applied,
            "hold_used": False,
            "reward": breakdown.as_dict(),
            "termination_reason": self.termination_reason,
        }
        return self.get_observation(), breakdown.total, self.done, info

    def _step_hold(self, action: Action) -> tuple[Observation, float, bool, dict]:
        previous_piece = self.current_piece
        if self.hold_piece is None:
            self.hold_piece = previous_piece
            self.current_piece = self.next_piece
            self.next_piece = self._randomizer.next_piece()
        else:
            self.current_piece, self.hold_piece = self.hold_piece, previous_piece
        self.can_hold = False

        if not self._actions_for_current_piece():
            self.done, self.termination_reason = True, "game_over"
        else:
            self.done, self.termination_reason = False, None
        self._last_metrics = calculate_metrics(self.board)
        breakdown = RewardBreakdown(0.0, 0.0, 0.0, 0.0, -10.0 if self.done else 0.0, -10.0 if self.done else 0.0)
        info = {
            "action": action,
            "lines_cleared": 0,
            "score_gain": 0,
            "level": self.level,
            "is_t_spin": False,
            "back_to_back_active": self.back_to_back_active,
            "back_to_back_bonus_applied": False,
            "hold_used": True,
            "held_piece": previous_piece,
            "reward": breakdown.as_dict(),
            "termination_reason": self.termination_reason,
        }
        return self.get_observation(), breakdown.total, self.done, info

    def _score_for_lock(self, lines_cleared: int, is_t_spin: bool) -> tuple[int, bool]:
        base_score = T_SPIN_SCORE[lines_cleared] if is_t_spin else LINE_CLEAR_SCORE[lines_cleared]
        b2b_eligible = lines_cleared > 0 and (is_t_spin or lines_cleared == 4)
        b2b_bonus_applied = b2b_eligible and self.back_to_back_active
        multiplier = B2B_MULTIPLIER if b2b_bonus_applied else 1.0
        if b2b_eligible:
            self.back_to_back_active = True
        elif lines_cleared > 0:
            self.back_to_back_active = False
        return int(base_score * self.level * multiplier), b2b_bonus_applied

    def _is_t_spin(self, piece: PieceType, shape: tuple[tuple[int, ...], ...], column: int, row: int) -> bool:
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
        return sum(1 for corner_row, corner_column in corners if self._is_filled_or_wall(corner_row, corner_column)) >= 3

    def _t_center(self, shape: tuple[tuple[int, ...], ...]) -> tuple[int, int] | None:
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
                occupied_neighbors = sum(1 for neighbor_row, neighbor_column in neighbors if self._shape_cell(shape, neighbor_row, neighbor_column))
                if occupied_neighbors >= 3:
                    return row, column
        return None

    def _shape_cell(self, shape: tuple[tuple[int, ...], ...], row: int, column: int) -> bool:
        return 0 <= row < len(shape) and 0 <= column < len(shape[row]) and bool(shape[row][column])

    def _is_filled_or_wall(self, row: int, column: int) -> bool:
        if column < 0 or column >= self.width or row < 0 or row >= self.height:
            return True
        return bool(self.board.matrix()[row][column])

    def clone(self) -> "TetrisEnv":
        clone = TetrisEnv.__new__(TetrisEnv)
        clone.width, clone.height, clone.max_pieces = self.width, self.height, self.max_pieces
        clone._initial_seed = self._initial_seed
        clone.board = self.board.clone()
        clone._randomizer = self._randomizer.clone()
        clone.current_piece, clone.next_piece = self.current_piece, self.next_piece
        clone.hold_piece = self.hold_piece
        clone.can_hold = self.can_hold
        clone.back_to_back_active = self.back_to_back_active
        clone.score = self.score
        clone.total_lines_cleared = self.total_lines_cleared
        clone.total_pieces_placed = self.total_pieces_placed
        clone.done, clone.termination_reason = self.done, self.termination_reason
        clone._last_metrics = self._last_metrics
        return clone
