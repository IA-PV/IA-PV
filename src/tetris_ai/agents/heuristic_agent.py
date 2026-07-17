from __future__ import annotations

from dataclasses import dataclass, field

from ..env.action import Action
from ..env.context import DecisionContext
from ..env.observation import Observation
from .base import Agent


@dataclass(frozen=True)
class TetrisGoal:
    """Objective function used by the state/goal/search agent."""

    name: str = "dellacherie_simplified"
    line_clear_weight: float = 3.4
    aggregate_height_penalty: float = 0.51
    hole_penalty: float = 35.6
    bumpiness_penalty: float = 0.18
    unsafe_height_threshold: int = 15
    unsafe_height_penalty: float = 20.0
    game_over_penalty: float = 10_000.0

    def evaluate_board(
        self,
        observation: Observation,
        termination_reason: str | None = None,
    ) -> float:
        metrics = observation.metrics
        if metrics is None:
            raise ValueError("StateGoalHeuristicAgent requires observation_mode='featured'.")
        unsafe_height = max(0, metrics.max_height - self.unsafe_height_threshold)
        value = (
            - self.aggregate_height_penalty * metrics.aggregate_height
            - self.hole_penalty * metrics.holes
            - self.bumpiness_penalty * metrics.bumpiness
            - self.unsafe_height_penalty * unsafe_height
        )
        reasons = set(termination_reason.split("+")) if termination_reason else set()
        if observation.terminated or "game_over" in reasons:
            value -= self.game_over_penalty
        return value


@dataclass(frozen=True)
class SearchResult:
    value: float
    plan: tuple[Action, ...]
    nodes_expanded: int


@dataclass(frozen=True)
class EvaluatedAction:
    value: float
    action: Action
    context: DecisionContext
    done: bool
    lines_cleared: int
    termination_reason: str | None


@dataclass
class AgentState:
    """Internal decision state for analysis and reporting."""

    decisions_made: int = 0
    nodes_expanded_total: int = 0
    max_nodes_single_decision: int = 0
    last_nodes_expanded: int = 0
    last_effective_depth: int = 0
    last_selected_value: float | None = None
    last_plan: tuple[Action, ...] = field(default_factory=tuple)
    last_observation: Observation | None = None

    @property
    def average_nodes_per_decision(self) -> float:
        if self.decisions_made == 0:
            return 0.0
        return self.nodes_expanded_total / self.decisions_made

    def record_decision(self, observation: Observation, result: SearchResult, effective_depth: int) -> None:
        self.decisions_made += 1
        self.nodes_expanded_total += result.nodes_expanded
        self.max_nodes_single_decision = max(self.max_nodes_single_decision, result.nodes_expanded)
        self.last_nodes_expanded = result.nodes_expanded
        self.last_effective_depth = effective_depth
        self.last_selected_value = result.value
        self.last_plan = result.plan
        self.last_observation = observation


class StateGoalHeuristicAgent(Agent):
    """Agent based on explicit state, goal, and heuristic search.

    The agent keeps an internal state from observations, defines a goal function
    over board quality, and searches future action sequences on cloned
    forward models containing only publicly visible pieces. The selected action
    is the first move of the best plan found.
    """

    def __init__(
        self,
        search_depth: int = 3,
        goal: TetrisGoal | None = None,
        beam_width: int = 8,
    ) -> None:
        if search_depth <= 0:
            raise ValueError("search_depth must be positive.")
        if beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        self.search_depth = search_depth
        self.goal = goal or TetrisGoal()
        self.beam_width = beam_width
        self.state = AgentState()

    def select_action(self, context: DecisionContext) -> Action:
        effective_depth = self._effective_search_depth(context.observation.level)
        result = self._search_best_plan(context, effective_depth)
        if not result.plan:
            raise RuntimeError("No legal action is available.")
        self.state.record_decision(context.observation, result, effective_depth)
        return result.plan[0]

    def _effective_search_depth(self, level: int) -> int:
        if level >= 10:
            return 1
        if level >= 5:
            return min(self.search_depth, 2)
        return self.search_depth

    def _search_best_plan(self, context: DecisionContext, depth: int) -> SearchResult:
        if depth <= 0 or context.observation.done:
            leaf_value = self.goal.evaluate_board(context.observation)
            return SearchResult(leaf_value, (), 0)

        evaluated_actions = self._rank_immediate_actions(context)
        if not evaluated_actions:
            leaf_value = self.goal.evaluate_board(context.observation)
            return SearchResult(leaf_value, (), 0)

        nodes_expanded = len(evaluated_actions)
        beam = evaluated_actions[: self.beam_width]

        if depth <= 1:
            best = beam[0]
            final_value = self.goal.evaluate_board(
                best.context.observation, best.termination_reason
            ) + (
                self.goal.line_clear_weight * best.lines_cleared
            )
            return SearchResult(final_value, (best.action,), nodes_expanded)

        best_result: SearchResult | None = None
        total_nodes_expanded = nodes_expanded

        for candidate in beam:
            if candidate.done:
                final_value = self.goal.evaluate_board(
                    candidate.context.observation, candidate.termination_reason
                ) + (
                    self.goal.line_clear_weight * candidate.lines_cleared
                )
                result = SearchResult(final_value, (candidate.action,), 0)
            else:
                future = self._search_best_plan(candidate.context, depth - 1)

                plan_value = self.goal.line_clear_weight * candidate.lines_cleared + future.value
                result = SearchResult(
                    plan_value,
                    (candidate.action, *future.plan),
                    future.nodes_expanded,
                )
                total_nodes_expanded += future.nodes_expanded

            if best_result is None or result.value > best_result.value:
                best_result = result

        assert best_result is not None
        return SearchResult(best_result.value, best_result.plan, total_nodes_expanded)

    def _rank_immediate_actions(self, context: DecisionContext) -> list[EvaluatedAction]:
        evaluated_actions = [
            self._evaluate_immediate_action(context, action) for action in context.legal_actions
        ]
        return sorted(evaluated_actions, key=lambda item: item.value, reverse=True)

    def _evaluate_immediate_action(self, context: DecisionContext, action: Action) -> EvaluatedAction:
        transition = context.simulate(action)
        observation = transition.context.observation
        lines_cleared = int(transition.info.get("lines_cleared", 0))
        termination_reason = transition.info.get("termination_reason")

        quick_estimate = self.goal.evaluate_board(
            observation, termination_reason
        ) + self.goal.line_clear_weight * lines_cleared

        return EvaluatedAction(
            quick_estimate,
            action,
            transition.context,
            transition.done,
            lines_cleared,
            termination_reason,
        )

    def decision_metrics(self) -> dict[str, int | float | str | None]:
        return {
            "goal": self.goal.name,
            "search_depth": self.search_depth,
            "effective_search_depth": self.state.last_effective_depth,
            "beam_width": self.beam_width,
            "decisions_made": self.state.decisions_made,
            "nodes_expanded": self.state.nodes_expanded_total,
            "avg_nodes_per_decision": self.state.average_nodes_per_decision,
            "max_nodes_single_decision": self.state.max_nodes_single_decision,
            "last_nodes_expanded": self.state.last_nodes_expanded,
            "last_selected_value": self.state.last_selected_value,
            "last_plan_length": len(self.state.last_plan),
        }


class HeuristicAgent(StateGoalHeuristicAgent):
    """Backward-compatible name for the state/goal/heuristic-search agent."""
