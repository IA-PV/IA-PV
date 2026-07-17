"""Numeric, UI-independent Tetris board operations."""

from __future__ import annotations

Shape = tuple[tuple[int, ...], ...]


class Board:
    """A rectangular board where ``0`` is empty and ``1`` is occupied."""

    def __init__(self, width: int = 10, height: int = 20) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Board dimensions must be positive.")
        self.width = width
        self.height = height
        self._grid: list[list[int]] = [[0] * width for _ in range(height)]

    @property
    def grid(self) -> tuple[tuple[int, ...], ...]:
        """An immutable snapshot of the board, safe to expose to callers."""
        return self.matrix()

    def matrix(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self._grid)

    def is_filled_or_wall(self, row: int, column: int) -> bool:
        if row < 0 or row >= self.height or column < 0 or column >= self.width:
            return True
        return bool(self._grid[row][column])

    def clone(self) -> "Board":
        clone = Board(self.width, self.height)
        clone._grid = [row.copy() for row in self._grid]
        return clone

    def is_valid_position(self, shape: Shape, column: int, row: int) -> bool:
        """Return whether all occupied cells fit without colliding.

        This planning ruleset has no hidden spawn rows. Every occupied cell must
        fit inside the visible board.
        """
        for shape_row, cells in enumerate(shape):
            for shape_column, occupied in enumerate(cells):
                if not occupied:
                    continue
                x, y = column + shape_column, row + shape_row
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    return False
                if self._grid[y][x]:
                    return False
        return True

    def drop_row(self, shape: Shape, column: int, start_row: int = 0) -> int | None:
        """Return the final row for a hard drop, or ``None`` if it cannot spawn."""
        if not self.is_valid_position(shape, column, start_row):
            return None
        row = start_row
        while self.is_valid_position(shape, column, row + 1):
            row += 1
        return row

    def place(self, shape: Shape, column: int, row: int) -> None:
        if not self.is_valid_position(shape, column, row):
            raise ValueError("Cannot place a piece in an invalid board position.")
        for shape_row, cells in enumerate(shape):
            for shape_column, occupied in enumerate(cells):
                if occupied:
                    y = row + shape_row
                    self._grid[y][column + shape_column] = 1

    def clear_lines(self) -> int:
        remaining = [row for row in self._grid if not all(row)]
        cleared = self.height - len(remaining)
        self._grid = [[0] * self.width for _ in range(cleared)] + remaining
        return cleared
