from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Action:
    rotation: int
    column: int


class InvalidActionError(ValueError):
    """Raised when an action is not currently legal."""


class EpisodeFinishedError(RuntimeError):
    """Raised when an agent attempts to step a completed episode."""
