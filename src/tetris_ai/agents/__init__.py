from typing import TYPE_CHECKING

from .base import Agent
from .heuristic_agent import HeuristicAgent, HeuristicSearchAgent, StateGoalHeuristicAgent, TetrisGoal
from .q_table_agent import QTableAgent
from .random_agent import RandomAgent
from .tetris_search_problem import (
    PiecePosition,
    SearchHeuristic,
    TetrisPlacementGoal,
    TetrisSearchProblem,
    TetrisSearchState,
)

# PyTorch is an optional dependency.  Keeping this import lazy preserves the
# lightweight core agents for installations that did not request the ``rl``
# extra, while still allowing ``from tetris_ai.agents import RLAgent``.
if TYPE_CHECKING:
    from .rl_agent import RLAgent, TetrisNet

__all__ = [
    "Agent",
    "HeuristicAgent",
    "HeuristicSearchAgent",
    "PiecePosition",
    "RandomAgent",
    "QTableAgent",
    "RLAgent",
    "SearchHeuristic",
    "StateGoalHeuristicAgent",
    "TetrisGoal",
    "TetrisPlacementGoal",
    "TetrisSearchProblem",
    "TetrisSearchState",
    "TetrisNet",
]


def __getattr__(name: str) -> object:
    if name in {"RLAgent", "TetrisNet"}:
        from .rl_agent import RLAgent, TetrisNet

        return {"RLAgent": RLAgent, "TetrisNet": TetrisNet}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
