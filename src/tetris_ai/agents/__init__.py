from typing import TYPE_CHECKING

from .base import Agent
from .genetic_agent import (
    GeneticAgent,
    GeneticModel,
    GeneticPolicyConfig,
    LinearChromosome,
    load_chromosome,
    load_genetic_model,
)
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
    "GeneticAgent",
    "GeneticModel",
    "GeneticPolicyConfig",
    "HeuristicAgent",
    "LinearChromosome",
    "QTableAgent",
    "RandomAgent",
    "RLAgent",
    "StateGoalHeuristicAgent",
    "TetrisGoal",
    "TetrisNet",
    "load_chromosome",
    "load_genetic_model",
]


def __getattr__(name: str) -> object:
    if name in {"RLAgent", "TetrisNet"}:
        from .rl_agent import RLAgent, TetrisNet

        return {"RLAgent": RLAgent, "TetrisNet": TetrisNet}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
