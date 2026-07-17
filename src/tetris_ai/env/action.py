from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Action:
    """Final placement, optionally holding before the piece is locked."""

    rotation: int = 0
    column: int = 0
    use_hold: bool = False

    @classmethod
    def hold(cls, rotation: int = 0, column: int = 0) -> "Action":
        return cls(rotation=rotation, column=column, use_hold=True)

    @property
    def is_hold(self) -> bool:
        return self.use_hold

    def to_id(self, board_width: int) -> int:
        if board_width <= 0:
            raise ValueError("board_width must be positive.")
        if not 0 <= self.rotation < 4 or not 0 <= self.column < board_width:
            raise ValueError("Action cannot be encoded for this board width.")
        hold_offset = 4 * board_width if self.use_hold else 0
        return hold_offset + self.rotation * board_width + self.column

    @classmethod
    def from_id(cls, action_id: int, board_width: int) -> "Action":
        action_space_size = 8 * board_width
        if board_width <= 0 or not 0 <= action_id < action_space_size:
            raise ValueError("action_id is outside the fixed action space.")
        use_hold = action_id >= 4 * board_width
        local_id = action_id % (4 * board_width)
        rotation, column = divmod(local_id, board_width)
        return cls(rotation=rotation, column=column, use_hold=use_hold)

    def as_dict(self, board_width: int) -> dict[str, int | bool]:
        return {
            "id": self.to_id(board_width),
            "rotation": self.rotation,
            "column": self.column,
            "use_hold": self.use_hold,
        }


class InvalidActionError(ValueError):
    """Raised when an action is not currently legal."""


class EpisodeFinishedError(RuntimeError):
    """Raised when an agent attempts to step a completed episode."""
