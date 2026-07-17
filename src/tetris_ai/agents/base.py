from __future__ import annotations

from abc import ABC, abstractmethod

from ..env.action import Action
from ..env.context import DecisionContext


class Agent(ABC):
    @abstractmethod
    def select_action(self, context: DecisionContext) -> Action:
        """Choose one action using only the public decision contract."""
