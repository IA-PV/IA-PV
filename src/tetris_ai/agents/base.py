from __future__ import annotations

from abc import ABC, abstractmethod

from ..env.action import Action
from ..env.tetris_env import TetrisEnv


class Agent(ABC):
    @abstractmethod
    def select_action(self, env: TetrisEnv) -> Action:
        """Choose one currently legal action."""
