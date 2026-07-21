"""Small deterministic episode evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor
from time import perf_counter

from ..agents.base import Agent
from ..env.tetris_env import TetrisEnv


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    agent: str
    score: int
    lines_removed: int
    pieces_placed: int
    total_reward: float
    termination_reason: str | None
    terminated: bool
    truncated: bool
    config_id: str
    task_return: float | None = None
    game_over: bool | None = None
    horizon_completed: bool | None = None
    policy_id: str | None = None
    search_depth: int = 0
    effective_search_depth: int = 0
    beam_width: int = 0
    decisions_made: int = 0
    nodes_expanded: int = 0
    avg_nodes_per_decision: float = 0.0
    max_nodes_single_decision: int = 0
    total_decision_time_seconds: float = 0.0
    mean_decision_time_ms: float = 0.0
    p50_decision_time_ms: float = 0.0
    p95_decision_time_ms: float = 0.0

    def __post_init__(self) -> None:
        """Fill fields absent from legacy, manually-created episode records."""

        reasons = _termination_reasons(self.termination_reason)
        if self.task_return is None:
            # Before task and shaped rewards were separated, total_reward was the
            # only persisted return. Keeping this fallback makes old fixtures and
            # report consumers readable without weakening the live evaluator.
            object.__setattr__(self, "task_return", self.total_reward)
        if self.game_over is None:
            object.__setattr__(self, "game_over", "game_over" in reasons)
        if self.horizon_completed is None:
            object.__setattr__(
                self,
                "horizon_completed",
                "horizon_completed" in reasons,
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_episode(agent: Agent, seed: int, max_pieces: int = 500) -> EpisodeResult:
    env = TetrisEnv(max_pieces=max_pieces, seed=seed)
    total_reward = 0.0
    task_return = 0.0
    decision_times_seconds: list[float] = []
    while not env.done:
        context = env.decision_context()
        started = perf_counter()
        action = agent.select_action(context)
        decision_times_seconds.append(perf_counter() - started)
        _, reward, _, _, info = env.step(action)
        total_reward += reward
        task_return += _task_reward_from(info, reward)
    decision_metrics = _decision_metrics_for(agent)
    total_decision_time = sum(decision_times_seconds)
    decision_times_ms = [duration * 1_000.0 for duration in decision_times_seconds]
    reasons = _termination_reasons(env.termination_reason)
    return EpisodeResult(
        seed=seed,
        agent=type(agent).__name__,
        score=env.score,
        lines_removed=env.total_lines_cleared,
        pieces_placed=env.total_pieces_placed,
        total_reward=total_reward,
        task_return=task_return,
        termination_reason=env.termination_reason,
        terminated=env.terminated,
        truncated=env.truncated,
        config_id=env.config.fingerprint,
        game_over="game_over" in reasons,
        horizon_completed="horizon_completed" in reasons,
        policy_id=decision_metrics["policy_id"],
        search_depth=decision_metrics["search_depth"],
        effective_search_depth=decision_metrics["effective_search_depth"],
        beam_width=decision_metrics["beam_width"],
        decisions_made=decision_metrics["decisions_made"],
        nodes_expanded=decision_metrics["nodes_expanded"],
        avg_nodes_per_decision=decision_metrics["avg_nodes_per_decision"],
        max_nodes_single_decision=decision_metrics["max_nodes_single_decision"],
        total_decision_time_seconds=total_decision_time,
        mean_decision_time_ms=(
            total_decision_time * 1_000.0 / len(decision_times_seconds)
            if decision_times_seconds
            else 0.0
        ),
        p50_decision_time_ms=_quantile(decision_times_ms, 0.50),
        p95_decision_time_ms=_quantile(decision_times_ms, 0.95),
    )


def _task_reward_from(info: object, fallback_reward: float) -> float:
    """Read the unshaped task reward while tolerating pre-v3 environments."""

    if isinstance(info, dict):
        reward = info.get("reward")
        if isinstance(reward, dict):
            task_reward = reward.get("task_reward")
            if isinstance(task_reward, (int, float)):
                return float(task_reward)
            # Legacy RewardBreakdown exposed the clean line utility under this
            # name. It is a safe compatibility fallback, unlike total reward.
            line_reward = reward.get("line_reward")
            if isinstance(line_reward, (int, float)):
                return float(line_reward)
    return float(fallback_reward)


def _termination_reasons(reason: str | None) -> frozenset[str]:
    if not reason:
        return frozenset()
    return frozenset(part for part in reason.split("+") if part)


def _quantile(values: list[float], probability: float) -> float:
    """Return a deterministic linearly interpolated sample quantile."""

    if not values:
        return 0.0
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _decision_metrics_for(agent: Agent) -> dict[str, int | float | str | None]:
    metrics_method = getattr(agent, "decision_metrics", None)
    if not callable(metrics_method):
        return _empty_decision_metrics()
    raw_metrics = metrics_method()
    return {
        "policy_id": str(raw_metrics["chromosome_id"])
        if raw_metrics.get("chromosome_id")
        else None,
        "search_depth": int(raw_metrics.get("search_depth", 0) or 0),
        "effective_search_depth": int(raw_metrics.get("effective_search_depth", 0) or 0),
        "beam_width": int(raw_metrics.get("beam_width", 0) or 0),
        "decisions_made": int(raw_metrics.get("decisions_made", 0) or 0),
        "nodes_expanded": int(raw_metrics.get("nodes_expanded", 0) or 0),
        "avg_nodes_per_decision": float(raw_metrics.get("avg_nodes_per_decision", 0.0) or 0.0),
        "max_nodes_single_decision": int(raw_metrics.get("max_nodes_single_decision", 0) or 0),
    }


def _empty_decision_metrics() -> dict[str, int | float | str | None]:
    return {
        "policy_id": None,
        "search_depth": 0,
        "effective_search_depth": 0,
        "beam_width": 0,
        "decisions_made": 0,
        "nodes_expanded": 0,
        "avg_nodes_per_decision": 0.0,
        "max_nodes_single_decision": 0,
    }
