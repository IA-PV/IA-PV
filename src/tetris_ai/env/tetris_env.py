"""Turn-based, headless Tetris environment."""

from __future__ import annotations

from ..core.board import Board
from ..core.metrics import BoardMetrics, calculate_metrics
from ..core.randomizer import SevenBagRandomizer
from ..core.tetromino import PieceType, rotations_for
from .action import Action, EpisodeFinishedError, InvalidActionError
from .observation import Observation
from .reward import calculate_reward

LINE_CLEAR_SCORE = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}


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
        self.score = 0
        self.total_lines_cleared = 0
        self.total_pieces_placed = 0
        self.done = not self._actions_for_current_piece()
        self.termination_reason: str | None = "game_over" if self.done else None
        self._last_metrics = calculate_metrics(self.board)
        return self.get_observation()

    def _actions_for_current_piece(self) -> tuple[Action, ...]:
        actions: list[Action] = []
        for rotation, shape in enumerate(rotations_for(self.current_piece)):
            shape_width = len(shape[0])
            for column in range(self.width - shape_width + 1):
                if self.board.drop_row(shape, column) is not None:
                    actions.append(Action(rotation, column))
        return tuple(actions)

    def legal_actions(self) -> tuple[Action, ...]:
        return () if self.done else self._actions_for_current_piece()

    def get_observation(self) -> Observation:
        return Observation(
            board=self.board.matrix(),
            current_piece=self.current_piece,
            next_piece=self.next_piece,
            score=self.score,
            total_lines_cleared=self.total_lines_cleared,
            total_pieces_placed=self.total_pieces_placed,
            done=self.done,
            metrics=self._last_metrics,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self.done:
            raise EpisodeFinishedError("Cannot step a finished Tetris episode; call reset().")
        if action not in self.legal_actions():
            raise InvalidActionError(f"Action {action!r} is not legal for {self.current_piece.value}.")
        before = calculate_metrics(self.board)
        shape = rotations_for(self.current_piece)[action.rotation]
        row = self.board.drop_row(shape, action.column)
        assert row is not None  # guaranteed by legal_actions
        self.board.place(shape, action.column, row)
        lines_cleared = self.board.clear_lines()
        self.score += LINE_CLEAR_SCORE[lines_cleared]
        self.total_lines_cleared += lines_cleared
        self.total_pieces_placed += 1

        self.current_piece = self.next_piece
        self.next_piece = self._randomizer.next_piece()
        if self.total_pieces_placed >= self.max_pieces:
            self.done, self.termination_reason = True, "piece_limit"
        elif not self._actions_for_current_piece():
            self.done, self.termination_reason = True, "game_over"
        else:
            self.done, self.termination_reason = False, None

        self._last_metrics = calculate_metrics(self.board, lines_cleared)
        breakdown = calculate_reward(before, self._last_metrics, lines_cleared, self.done)
        info = {
            "action": action,
            "lines_cleared": lines_cleared,
            "reward": breakdown.as_dict(),
            "termination_reason": self.termination_reason,
        }
        return self.get_observation(), breakdown.total, self.done, info

    def clone(self) -> "TetrisEnv":
        clone = TetrisEnv.__new__(TetrisEnv)
        clone.width, clone.height, clone.max_pieces = self.width, self.height, self.max_pieces
        clone._initial_seed = self._initial_seed
        clone.board = self.board.clone()
        clone._randomizer = self._randomizer.clone()
        clone.current_piece, clone.next_piece = self.current_piece, self.next_piece
        clone.score = self.score
        clone.total_lines_cleared = self.total_lines_cleared
        clone.total_pieces_placed = self.total_pieces_placed
        clone.done, clone.termination_reason = self.done, self.termination_reason
        clone._last_metrics = self._last_metrics
        return clone
