from __future__ import annotations

from abc import ABC, abstractmethod

from ..env.action import Action
from ..env.context import DecisionContext


class Agent(ABC):
    def configuration(self) -> dict[str, object]:
        """Return serializable policy settings used for experiment reporting.

        Stateless agents may keep the default empty mapping.  Implementations
        should expose constructor-level configuration here, not mutable episode
        telemetry (which belongs in ``decision_metrics``).
        """

        return {}

    @abstractmethod
    def select_action(self, context: DecisionContext) -> Action:
        """Choose one action using only the public decision contract."""
