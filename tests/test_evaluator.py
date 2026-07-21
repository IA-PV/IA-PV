from types import SimpleNamespace

import pytest

from tetris_ai.agents.base import Agent
from tetris_ai.evaluation import evaluator


class _SingleDecisionAgent(Agent):
    def select_action(self, context):
        assert context == "decision-context"
        return "selected-action"


class _SingleStepEnvironment:
    def __init__(self, max_pieces: int, seed: int) -> None:
        assert max_pieces == 7
        assert seed == 11
        self.done = False
        self.score = 300
        self.total_lines_cleared = 2
        self.total_pieces_placed = 1
        self.termination_reason = None
        self.terminated = False
        self.truncated = False
        self.config = SimpleNamespace(fingerprint="test-config")

    def decision_context(self):
        return "decision-context"

    def step(self, action):
        assert action == "selected-action"
        self.done = True
        self.termination_reason = "game_over+horizon_completed"
        self.terminated = True
        return (
            None,
            99.0,
            True,
            False,
            {"reward": {"task_reward": 3.0, "total": 99.0}},
        )


def test_episode_separates_task_return_and_measures_only_decision_time(
    monkeypatch,
) -> None:
    clock = iter((10.0, 10.025))
    monkeypatch.setattr(evaluator, "TetrisEnv", _SingleStepEnvironment)
    monkeypatch.setattr(evaluator, "perf_counter", lambda: next(clock))

    result = evaluator.evaluate_episode(_SingleDecisionAgent(), seed=11, max_pieces=7)

    assert result.task_return == 3.0
    assert result.total_reward == 99.0
    assert result.game_over is True
    assert result.horizon_completed is True
    assert result.total_decision_time_seconds == pytest.approx(0.025)
    assert result.mean_decision_time_ms == pytest.approx(25.0)
    assert result.p50_decision_time_ms == pytest.approx(25.0)
    assert result.p95_decision_time_ms == pytest.approx(25.0)


def test_quantile_uses_deterministic_linear_interpolation() -> None:
    assert evaluator._quantile([], 0.95) == 0.0
    assert evaluator._quantile([40.0, 10.0, 20.0, 30.0], 0.50) == 25.0
    assert evaluator._quantile([0.0, 100.0], 0.95) == 95.0
