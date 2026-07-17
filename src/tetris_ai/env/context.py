"""Agent-facing decision contract with no access to the live environment RNG."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .action import Action
from .observation import Observation
from .telemetry import StepInfo


@dataclass(frozen=True)
class SimulatedTransition:
    context: "DecisionContext"
    reward: float
    terminated: bool
    truncated: bool
    info: StepInfo

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


Simulator = Callable[[Action], SimulatedTransition]


@dataclass(frozen=True)
class DecisionContext:
    """Everything an agent may use while choosing an action."""

    observation: Observation
    legal_actions: tuple[Action, ...]
    _simulator: Simulator = field(repr=False, compare=False)

    def simulate(self, action: Action) -> SimulatedTransition:
        if action not in self.legal_actions:
            raise ValueError(f"Action {action!r} is not legal in this decision context.")
        return self._simulator(action)
