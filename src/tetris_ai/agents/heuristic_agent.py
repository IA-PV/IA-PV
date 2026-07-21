"""Heuristic best-first agents built on an explicit Tetris search problem."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import heapq
import itertools
import math
from typing import Literal

from ..env.action import Action
from ..env.context import DecisionContext
from ..env.observation import Observation
from .base import Agent
from .tetris_search_problem import (
    TetrisPlacementGoal,
    TetrisSearchProblem,
    TetrisSearchState,
)


HeuristicFunction = Callable[[TetrisSearchState], float]
SearchStrategy = Literal["astar", "greedy"]


@dataclass(frozen=True)
class TetrisGoal:
    """Board-cost heuristic used to judge a locked Tetris placement.

    Tetris has no global winning state, so the practical objective is local:
    among the final placements of the current piece (and visible lookahead
    pieces), prefer the board with the smallest penalty.  The cost intentionally
    follows the teaching formulation ``holes * weight + aggregate_height *
    weight``.  A game-over successor receives a large additional cost.
    """

    name: str = "minimize_holes_and_aggregate_height"
    aggregate_height_penalty: float = 0.51
    hole_penalty: float = 35.6
    game_over_penalty: float = 10_000.0

    def __post_init__(self) -> None:
        if self.aggregate_height_penalty < 0 or self.hole_penalty < 0:
            raise ValueError("Heuristic penalty weights must be non-negative.")
        if self.game_over_penalty < 0:
            raise ValueError("game_over_penalty must be non-negative.")

    def evaluate_board(
        self,
        observation: Observation,
        termination_reason: str | None = None,
    ) -> float:
        """Return a penalty cost; lower values describe better boards."""

        metrics = observation.metrics
        if metrics is None:
            raise ValueError("HeuristicSearchAgent requires observation_mode='featured'.")
        game_over = observation.terminated or self._has_game_over(termination_reason)
        return self._evaluate_metrics(metrics.aggregate_height, metrics.holes, game_over)

    def evaluate_state(self, state: TetrisSearchState) -> float:
        """Evaluate a search state without looking past its public preview."""

        metrics = state.metrics
        if metrics is None:
            raise ValueError("HeuristicSearchAgent requires observation_mode='featured'.")
        return self._evaluate_metrics(metrics.aggregate_height, metrics.holes, state.terminated)

    def _evaluate_metrics(self, aggregate_height: int, holes: int, game_over: bool) -> float:
        cost = (
            self.aggregate_height_penalty * aggregate_height
            + self.hole_penalty * holes
        )
        if game_over:
            cost += self.game_over_penalty
        return cost

    @staticmethod
    def _has_game_over(termination_reason: str | None) -> bool:
        return bool(termination_reason and "game_over" in termination_reason.split("+"))


@dataclass(frozen=True)
class SearchResult:
    """Plan returned by a best-first search, ordered from the root action."""

    cost: float
    plan: tuple[Action, ...]
    nodes_expanded: int
    goal_reached: bool = True

    @property
    def value(self) -> float:
        """Backward-compatible alias for the selected objective value.

        The value is a *cost* in this refactoring, so lower is better.
        """

        return self.cost


@dataclass(frozen=True)
class SearchNode:
    """A node in the explicitly modelled state-space search tree."""

    state: TetrisSearchState
    plan: tuple[Action, ...]
    path_cost: int
    heuristic_cost: float

    @property
    def f_cost(self) -> float:
        return self.path_cost + self.heuristic_cost


@dataclass
class AgentState:
    """Internal state retained between calls to :meth:`act`."""

    decisions_made: int = 0
    nodes_expanded_total: int = 0
    max_nodes_single_decision: int = 0
    last_nodes_expanded: int = 0
    last_effective_depth: int = 0
    last_selected_cost: float | None = None
    # Kept for existing telemetry consumers.  It now contains a cost, not a
    # maximisation score.
    last_selected_value: float | None = None
    last_plan: tuple[Action, ...] = field(default_factory=tuple)
    last_observation: Observation | None = None

    @property
    def average_nodes_per_decision(self) -> float:
        if self.decisions_made == 0:
            return 0.0
        return self.nodes_expanded_total / self.decisions_made

    def record_search(
        self,
        observation: Observation,
        result: SearchResult,
        effective_depth: int,
    ) -> None:
        self.decisions_made += 1
        self.nodes_expanded_total += result.nodes_expanded
        self.max_nodes_single_decision = max(self.max_nodes_single_decision, result.nodes_expanded)
        self.last_nodes_expanded = result.nodes_expanded
        self.last_effective_depth = effective_depth
        self.last_selected_cost = result.cost
        self.last_selected_value = result.cost
        self.last_plan = result.plan
        self.last_observation = observation

    def record_plan_execution(self, observation: Observation, remaining_plan: list[Action]) -> None:
        """Record an action taken from an already computed internal plan."""

        self.decisions_made += 1
        self.last_nodes_expanded = 0
        self.last_plan = tuple(remaining_plan)
        self.last_observation = observation


class HeuristicSearchAgent(Agent):
    """A stateful greedy best-first (or A*) Tetris placement agent.

    A call to :meth:`act` first consumes an action from the internal plan.  If
    no valid plan remains, it formulates a placement goal and a
    :class:`TetrisSearchProblem`, runs best-first search, then executes the
    first planned final placement.

    ``max_nodes_expanded`` is a planning budget, not a beam: nodes are always
    selected from one global priority queue.  It prevents a deep public
    lookahead from growing exponentially on a nearly empty board.  Set it to
    ``None`` to require exhaustive best-first search.  ``beam_width`` is
    accepted only for source compatibility with the previous beam-search
    agent; it does not prune this search tree.
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction | None = None,
        *,
        search_depth: int = 3,
        goal: TetrisGoal | None = None,
        search_strategy: SearchStrategy = "greedy",
        max_nodes_expanded: int | None = 2_000,
        beam_width: int | None = None,
    ) -> None:
        if search_depth <= 0:
            raise ValueError("search_depth must be positive.")
        if search_strategy not in ("astar", "greedy"):
            raise ValueError("search_strategy must be 'astar' or 'greedy'.")
        if max_nodes_expanded is not None and max_nodes_expanded <= 0:
            raise ValueError("max_nodes_expanded must be positive when provided.")
        if beam_width is not None and beam_width <= 0:
            raise ValueError("beam_width must be positive when provided.")

        self.search_depth = search_depth
        self.goal = goal or TetrisGoal()
        self.heuristic: HeuristicFunction = heuristic_function or self.goal.evaluate_state
        self.search_strategy = search_strategy
        self.max_nodes_expanded = max_nodes_expanded
        self.beam_width = beam_width
        self.plan: list[Action] = []
        self.state = AgentState()

    def act(self, state: DecisionContext) -> Action:
        """Choose the next final placement using retained plan state when safe."""

        if self.plan:
            planned_action = self.plan[0]
            if planned_action in state.legal_actions:
                self.plan.pop(0)
                self.state.record_plan_execution(state.observation, self.plan)
                return planned_action
            # A context can be externally replaced between decisions.  Never
            # execute a placement planned for a different public state.
            self.plan.clear()

        goal = self.formulate_goal(state)
        problem = self.formulate_problem(state, goal)
        result = self.search(problem)
        self.plan = list(result.plan)
        self.state.record_search(state.observation, result, goal.pieces_to_lock)

        if self.plan:
            return self.plan.pop(0)
        if state.legal_actions:
            # In this final-placement API there is no separate FALL command;
            # the deterministic safe fallback is a legal hard-drop placement.
            return state.legal_actions[0]
        raise RuntimeError("No legal action is available.")

    def select_action(self, context: DecisionContext) -> Action:
        """Environment-compatible name for :meth:`act`."""

        return self.act(context)

    def formulate_goal(self, context: DecisionContext) -> TetrisPlacementGoal:
        """Lock the current piece and as much visible lookahead as requested."""

        visible_horizon = 1 + len(context.observation.next_pieces)
        depth = min(self._effective_search_depth(context.observation.level), visible_horizon)
        return TetrisPlacementGoal(pieces_to_lock=depth)

    def formulate_problem(
        self,
        context: DecisionContext,
        goal: TetrisPlacementGoal,
    ) -> TetrisSearchProblem:
        """Create the explicit state/action/cost problem for this decision."""

        return TetrisSearchProblem(TetrisSearchState.initial(context), goal, self.heuristic)

    def search(self, problem: TetrisSearchProblem) -> SearchResult:
        """Run A* or greedy best-first search over final placements.

        For A*, the priority is ``f(n) = g(n) + h(n)``.  For greedy search it
        is ``f(n) = h(n)``.  ``g(n)`` counts locked placements, and ``h(n)`` is
        the board penalty produced by ``heuristic_function``.
        """

        root = SearchNode(
            state=problem.initial_state,
            plan=(),
            path_cost=0,
            heuristic_cost=self._heuristic_cost(problem.heuristic(problem.initial_state)),
        )
        frontier: list[tuple[float, int, SearchNode]] = []
        sequence = itertools.count()
        heapq.heappush(frontier, (self._priority(root), next(sequence), root))
        nodes_expanded = 0
        best_partial = root

        while frontier:
            priority, _, node = heapq.heappop(frontier)
            if self._is_better_partial(node, priority, best_partial):
                best_partial = node
            if problem.is_goal(node.state, node.path_cost):
                return SearchResult(node.f_cost, node.plan, nodes_expanded, goal_reached=True)

            for action in problem.actions(node.state):
                if self.max_nodes_expanded is not None and nodes_expanded >= self.max_nodes_expanded:
                    return SearchResult(
                        best_partial.f_cost,
                        best_partial.plan,
                        nodes_expanded,
                        goal_reached=False,
                    )
                successor = problem.result(node.state, action)
                path_cost = problem.path_cost(node.path_cost, action, successor)
                child = SearchNode(
                    state=successor,
                    plan=(*node.plan, action),
                    path_cost=path_cost,
                    heuristic_cost=self._heuristic_cost(problem.heuristic(successor)),
                )
                nodes_expanded += 1
                heapq.heappush(frontier, (self._priority(child), next(sequence), child))

        # This occurs only for a terminal root with no legal placement.  The
        # caller turns it into a clear "no legal action" error if necessary.
        return SearchResult(root.f_cost, (), nodes_expanded, goal_reached=False)

    def _effective_search_depth(self, level: int) -> int:
        """Reduce lookahead at high speed while preserving a legal minimum."""

        if level >= 10:
            return 1
        if level >= 5:
            return min(self.search_depth, 2)
        return self.search_depth

    def _heuristic_cost(self, value: float) -> float:
        cost = float(value)
        if not math.isfinite(cost):
            raise ValueError("heuristic_function must return a finite cost.")
        return cost

    def _priority(self, node: SearchNode) -> float:
        if self.search_strategy == "greedy":
            return node.heuristic_cost
        return node.f_cost

    def _is_better_partial(
        self,
        candidate: SearchNode,
        candidate_priority: float,
        current: SearchNode,
    ) -> bool:
        """Prefer a deeper partial plan, then the lower queue priority."""

        if candidate.path_cost != current.path_cost:
            return candidate.path_cost > current.path_cost
        return candidate_priority < self._priority(current)

    def decision_metrics(self) -> dict[str, int | float | str | None]:
        """Expose search diagnostics to the evaluator and visualizer."""

        return {
            "goal": self.goal.name,
            "search_strategy": self.search_strategy,
            "max_nodes_expanded": self.max_nodes_expanded,
            "search_depth": self.search_depth,
            "effective_search_depth": self.state.last_effective_depth,
            # Compatibility telemetry; it does not control A*/greedy search.
            "beam_width": self.beam_width or 0,
            "decisions_made": self.state.decisions_made,
            "nodes_expanded": self.state.nodes_expanded_total,
            "avg_nodes_per_decision": self.state.average_nodes_per_decision,
            "max_nodes_single_decision": self.state.max_nodes_single_decision,
            "last_nodes_expanded": self.state.last_nodes_expanded,
            "last_selected_cost": self.state.last_selected_cost,
            "last_selected_value": self.state.last_selected_value,
            "last_plan_length": len(self.state.last_plan),
        }


class StateGoalHeuristicAgent(HeuristicSearchAgent):
    """Backward-compatible name for :class:`HeuristicSearchAgent`."""


class HeuristicAgent(HeuristicSearchAgent):
    """Short backward-compatible name for the heuristic search agent."""


__all__ = [
    "AgentState",
    "HeuristicAgent",
    "HeuristicFunction",
    "HeuristicSearchAgent",
    "SearchResult",
    "SearchStrategy",
    "StateGoalHeuristicAgent",
    "TetrisGoal",
]
