from __future__ import annotations

from ..env.action import Action
from ..env.tetris_env import TetrisEnv
from .base import Agent


class HeuristicAgent(Agent):
    def select_action(self, env: TetrisEnv) -> Action:
        best_action: Action | None = None
        best_value: float | None = None
        for action in env.legal_actions():
            observation, _, _, _ = env.clone().step(action)
            metrics = observation.metrics
            value = (
                1.0 * metrics.lines_cleared_last_move
                - 0.5 * metrics.aggregate_height
                - 4.0 * metrics.holes
                - 0.8 * metrics.bumpiness
            )
            if best_value is None or value > best_value:
                best_action, best_value = action, value
        if best_action is None:
            raise RuntimeError("No legal action is available.")
        return best_action
