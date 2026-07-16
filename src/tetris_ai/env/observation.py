from dataclasses import dataclass

from ..core.metrics import BoardMetrics
from ..core.tetromino import PieceType


@dataclass(frozen=True)
class Observation:
    board: tuple[tuple[int, ...], ...]
    current_piece: PieceType
    next_piece: PieceType
    hold_piece: PieceType | None
    can_hold: bool
    score: int
    level: int
    total_lines_cleared: int
    total_pieces_placed: int
    back_to_back_active: bool
    done: bool
    metrics: BoardMetrics
