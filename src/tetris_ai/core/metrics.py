"""Features used by observations, rewards, and simple agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .board import Board


@dataclass(frozen=True)
class BoardMetrics:
    column_heights: tuple[int, ...]
    aggregate_height: int
    max_height: int
    holes: int
    bumpiness: int
    lines_cleared_last_move: int = 0
    # Dellacherie-style surface features.  Defaulted so hand-built metrics in
    # tests and legacy fixtures remain valid; ``calculate_metrics`` always fills
    # them from the board.
    row_transitions: int = 0
    column_transitions: int = 0
    wells: int = 0

    def as_dict(self) -> dict[str, int | tuple[int, ...]]:
        return asdict(self)


def calculate_metrics(board: Board, lines_cleared_last_move: int = 0) -> BoardMetrics:
    grid = board.matrix()
    width = board.width
    height = board.height
    heights: list[int] = []
    holes = 0
    column_transitions = 0
    for column in range(width):
        first_filled = next((row for row in range(height) if grid[row][column]), height)
        heights.append(height - first_filled)
        holes += sum(1 for row in range(first_filled, height) if not grid[row][column])
        # Vertical filled<->empty flips.  The cell above the board is empty and
        # the floor below it is filled (Dellacherie convention).
        previous_filled = False
        for row in range(height):
            filled = bool(grid[row][column])
            if filled != previous_filled:
                column_transitions += 1
            previous_filled = filled
        if not previous_filled:
            column_transitions += 1

    bumpiness = sum(abs(left - right) for left, right in zip(heights, heights[1:]))

    # Horizontal filled<->empty flips, counting both side walls as filled.
    row_transitions = 0
    for row in range(height):
        previous_filled = True
        for column in range(width):
            filled = bool(grid[row][column])
            if filled != previous_filled:
                row_transitions += 1
            previous_filled = filled
        if not previous_filled:
            row_transitions += 1

    # Aggregate well depth: a cell is "in a well" when both neighbours (or the
    # side walls, treated as full-height) are taller than its column.
    wells = 0
    last_column = width - 1
    for column, column_height in enumerate(heights):
        left = height if column == 0 else heights[column - 1]
        right = height if column == last_column else heights[column + 1]
        wells += max(0, min(left, right) - column_height)

    return BoardMetrics(
        column_heights=tuple(heights),
        aggregate_height=sum(heights),
        max_height=max(heights, default=0),
        holes=holes,
        bumpiness=bumpiness,
        lines_cleared_last_move=lines_cleared_last_move,
        row_transitions=row_transitions,
        column_transitions=column_transitions,
        wells=wells,
    )
