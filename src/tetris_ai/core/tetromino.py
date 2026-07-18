"""The seven tetrominoes and their precomputed unique rotations."""

from __future__ import annotations

from enum import Enum

from .board import Shape


class PieceType(str, Enum):
    I = "I"
    J = "J"
    L = "L"
    O = "O"
    S = "S"
    T = "T"
    Z = "Z"


def _trim(shape: Shape) -> Shape:
    rows = [row for row in shape if any(row)]
    left = min(index for row in rows for index, value in enumerate(row) if value)
    right = max(index for row in rows for index, value in enumerate(row) if value)
    return tuple(tuple(row[left : right + 1]) for row in rows)


def _rotate_clockwise(shape: Shape) -> Shape:
    return _trim(tuple(tuple(row) for row in zip(*shape[::-1])))


def _unique_rotations(shape: Shape) -> tuple[Shape, ...]:
    rotations: list[Shape] = []
    rotated = _trim(shape)
    for _ in range(4):
        if rotated not in rotations:
            rotations.append(rotated)
        rotated = _rotate_clockwise(rotated)
    return tuple(rotations)


_BASE_SHAPES: dict[PieceType, Shape] = {
    PieceType.I: ((1, 1, 1, 1),),
    PieceType.J: ((1, 0, 0), (1, 1, 1)),
    PieceType.L: ((0, 0, 1), (1, 1, 1)),
    PieceType.O: ((1, 1), (1, 1)),
    PieceType.S: ((0, 1, 1), (1, 1, 0)),
    PieceType.T: ((0, 1, 0), (1, 1, 1)),
    PieceType.Z: ((1, 1, 0), (0, 1, 1)),
}

ROTATIONS: dict[PieceType, tuple[Shape, ...]] = {
    piece: _unique_rotations(shape) for piece, shape in _BASE_SHAPES.items()
}


def rotations_for(piece: PieceType) -> tuple[Shape, ...]:
    return ROTATIONS[piece]
