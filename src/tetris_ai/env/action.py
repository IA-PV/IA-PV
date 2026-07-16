from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Action:
    rotation: int = 0
    column: int = 0
    use_hold: bool = False

    @classmethod
    def hold(cls) -> "Action":
        return cls(rotation=0, column=0, use_hold=True)

    @property
    def is_hold(self) -> bool:
        return self.use_hold


class InvalidActionError(ValueError):
    """Raised when an action is not currently legal."""


class EpisodeFinishedError(RuntimeError):
    """Raised when an agent attempts to step a completed episode."""
