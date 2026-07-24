"""Numeric, UI-independent Tetris board operations."""

from __future__ import annotations

Shape = tuple[tuple[int, ...], ...]


# Placement checks only care about a shape's filled cells, but the raw matrix is
# mostly empty padding.  Rotations come from a small fixed set of shared tuples,
# so caching each shape's filled (row, column) offsets turns the hot collision
# loop from "whole bounding box" into "the four occupied cells".
_FILLED_CELLS: dict[Shape, tuple[tuple[int, int], ...]] = {}


def _filled_cells(shape: Shape) -> tuple[tuple[int, int], ...]:
    cells = _FILLED_CELLS.get(shape)
    if cells is None:
        cells = tuple(
            (row, column)
            for row, cells_in_row in enumerate(shape)
            for column, occupied in enumerate(cells_in_row)
            if occupied
        )
        _FILLED_CELLS[shape] = cells
    return cells


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
        grid = self._grid
        width = self.width
        height = self.height
        for shape_row, shape_column in _filled_cells(shape):
            x = column + shape_column
            y = row + shape_row
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
            if grid[y][x]:
                return False
        return True

    def drop_row(self, shape: Shape, column: int, start_row: int = 0) -> int | None:
        """Return the final row for a hard drop, or ``None`` if it cannot spawn.

        Semantically identical to repeatedly calling :meth:`is_valid_position`
        one row at a time, but it scans only the shape's filled cells against the
        live grid, which is the dominant cost during search-based planning.
        """
        cells = _filled_cells(shape)
        grid = self._grid
        width = self.width
        height = self.height

        def fits(candidate_row: int) -> bool:
            for shape_row, shape_column in cells:
                x = column + shape_column
                y = candidate_row + shape_row
                if x < 0 or x >= width or y < 0 or y >= height:
                    return False
                if grid[y][x]:
                    return False
            return True

        if not fits(start_row):
            return None
        row = start_row
        while fits(row + 1):
            row += 1
        return row

    def next_filled_below(self) -> tuple[tuple[int, ...], ...]:
        """Per column, the smallest filled row at or below each row (else height).

        ``result[column][row]`` is the first filled cell a piece cell starting at
        ``row`` would meet as it falls.  Indexable at ``row == height`` so callers
        can look one past the bottom without a branch.
        """
        height = self.height
        grid = self._grid
        columns: list[tuple[int, ...]] = []
        for column in range(self.width):
            column_map = [height] * (height + 1)
            first_filled = height
            for row in range(height - 1, -1, -1):
                if grid[row][column]:
                    first_filled = row
                column_map[row] = first_filled
            columns.append(tuple(column_map))
        return tuple(columns)

    def hard_drop_row(
        self,
        shape: Shape,
        column: int,
        next_filled_below: tuple[tuple[int, ...], ...],
    ) -> int | None:
        """Landing row for a hard drop, or ``None`` if the piece cannot spawn.

        Bit-for-bit equivalent to :meth:`drop_row` but ``O(cells)`` given a
        ``next_filled_below`` map, so enumerating every placement on one board is
        ``O(cells)`` per candidate instead of a full column scan.  Each filled
        cell falls until the first filled board cell beneath it; the whole piece
        stops at the tightest of those limits.
        """
        landing = self.height
        for shape_row, shape_column in _filled_cells(shape):
            board_column = column + shape_column
            column_map = next_filled_below[board_column]
            if column_map[shape_row] == shape_row:
                return None  # A filled cell blocks the spawn position itself.
            drop = column_map[shape_row + 1] - shape_row - 1
            if drop < landing:
                landing = drop
        return landing

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
