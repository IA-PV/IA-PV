from dataclasses import dataclass

from ..core.metrics import BoardMetrics
from ..core.tetromino import PieceType


@dataclass(frozen=True)
class Observation:
    board: tuple[tuple[int, ...], ...]
    current_piece: PieceType | None
    next_pieces: tuple[PieceType, ...]
    hold_piece: PieceType | None
    can_hold: bool
    score: int
    level: int
    total_lines_cleared: int
    total_pieces_placed: int
    back_to_back_active: bool
    terminated: bool
    truncated: bool
    action_mask: tuple[bool, ...]
    metrics: BoardMetrics | None

    @property
    def next_piece(self) -> PieceType | None:
        return self.next_pieces[0] if self.next_pieces else None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
