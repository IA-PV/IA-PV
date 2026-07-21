"""Small deterministic episode evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
    search_depth: int = 0
    effective_search_depth: int = 0
    search_strategy: str = ""
    max_nodes_expanded: int = 0
    # Retained in the exported schema for historical CSV compatibility.  The
    # heuristic search no longer uses a beam width.
    beam_width: int = 0
    decisions_made: int = 0
    nodes_expanded: int = 0
    avg_nodes_per_decision: float = 0.0
    max_nodes_single_decision: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_episode(agent: Agent, seed: int, max_pieces: int = 500) -> EpisodeResult:
    env = TetrisEnv(max_pieces=max_pieces, seed=seed)
    total_reward = 0.0
    while not env.done:
        context = env.decision_context()
        _, reward, _, _, _ = env.step(agent.select_action(context))
        total_reward += reward
    decision_metrics = _decision_metrics_for(agent)
    return EpisodeResult(
        seed=seed,
        agent=type(agent).__name__,
        score=env.score,
        lines_removed=env.total_lines_cleared,
        pieces_placed=env.total_pieces_placed,
        total_reward=total_reward,
        termination_reason=env.termination_reason,
        terminated=env.terminated,
        truncated=env.truncated,
        config_id=env.config.fingerprint,
        search_depth=decision_metrics["search_depth"],
        effective_search_depth=decision_metrics["effective_search_depth"],
        search_strategy=decision_metrics["search_strategy"],
        max_nodes_expanded=decision_metrics["max_nodes_expanded"],
        beam_width=decision_metrics["beam_width"],
        decisions_made=decision_metrics["decisions_made"],
        nodes_expanded=decision_metrics["nodes_expanded"],
        avg_nodes_per_decision=decision_metrics["avg_nodes_per_decision"],
        max_nodes_single_decision=decision_metrics["max_nodes_single_decision"],
    )


def _decision_metrics_for(agent: Agent) -> dict[str, int | float | str]:
    metrics_method = getattr(agent, "decision_metrics", None)
    if not callable(metrics_method):
        return _empty_decision_metrics()
    raw_metrics = metrics_method()
    return {
        "search_depth": int(raw_metrics.get("search_depth", 0) or 0),
        "effective_search_depth": int(raw_metrics.get("effective_search_depth", 0) or 0),
        "search_strategy": str(raw_metrics.get("search_strategy", "") or ""),
        "max_nodes_expanded": int(raw_metrics.get("max_nodes_expanded", 0) or 0),
        "beam_width": int(raw_metrics.get("beam_width", 0) or 0),
        "decisions_made": int(raw_metrics.get("decisions_made", 0) or 0),
        "nodes_expanded": int(raw_metrics.get("nodes_expanded", 0) or 0),
        "avg_nodes_per_decision": float(raw_metrics.get("avg_nodes_per_decision", 0.0) or 0.0),
        "max_nodes_single_decision": int(raw_metrics.get("max_nodes_single_decision", 0) or 0),
    }


def _empty_decision_metrics() -> dict[str, int | float | str]:
    return {
        "search_depth": 0,
        "effective_search_depth": 0,
        "search_strategy": "",
        "max_nodes_expanded": 0,
        "beam_width": 0,
        "decisions_made": 0,
        "nodes_expanded": 0,
        "avg_nodes_per_decision": 0.0,
        "max_nodes_single_decision": 0,
    }
