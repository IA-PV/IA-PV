from typing import TYPE_CHECKING

from .base import Agent
from .heuristic_agent import HeuristicAgent, StateGoalHeuristicAgent, TetrisGoal
from .q_table_agent import QTableAgent
from .random_agent import RandomAgent

# PyTorch is an optional dependency.  Keeping this import lazy preserves the
# lightweight core agents for installations that did not request the ``rl``
# extra, while still allowing ``from tetris_ai.agents import RLAgent``.
if TYPE_CHECKING:
    from .rl_agent import RLAgent, TetrisNet

__all__ = [
    "Agent",
    "HeuristicAgent",
    "RandomAgent",
    "QTableAgent",
    "RLAgent",
    "StateGoalHeuristicAgent",
    "TetrisGoal",
    "TetrisNet",
]


def __getattr__(name: str) -> object:
    if name in {"RLAgent", "TetrisNet"}:
        from .rl_agent import RLAgent, TetrisNet

        return {"RLAgent": RLAgent, "TetrisNet": TetrisNet}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
