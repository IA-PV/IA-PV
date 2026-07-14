"""Features used by observations, rewards, and simple agents."""

from __future__ import annotations

from dataclasses import dataclass

from .board import Board


@dataclass(frozen=True)
class BoardMetrics:
    column_heights: tuple[int, ...]
    aggregate_height: int
    max_height: int
    holes: int
    bumpiness: int
    lines_cleared_last_move: int = 0


def calculate_metrics(board: Board, lines_cleared_last_move: int = 0) -> BoardMetrics:
    grid = board.matrix()
    heights: list[int] = []
    holes = 0
    for column in range(board.width):
        first_filled = next((row for row in range(board.height) if grid[row][column]), board.height)
        heights.append(board.height - first_filled)
        holes += sum(1 for row in range(first_filled, board.height) if not grid[row][column])
    bumpiness = sum(abs(left - right) for left, right in zip(heights, heights[1:]))
    return BoardMetrics(tuple(heights), sum(heights), max(heights, default=0), holes, bumpiness, lines_cleared_last_move)
