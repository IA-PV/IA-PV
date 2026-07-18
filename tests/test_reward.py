import pytest

from tetris_ai.core.metrics import BoardMetrics
from tetris_ai.env import RewardConfig
from tetris_ai.env.reward import calculate_reward


def test_reward_breakdown_uses_metric_deltas() -> None:
    before = BoardMetrics((2, 1), aggregate_height=3, max_height=2, holes=0, bumpiness=1)
    after = BoardMetrics((4, 1), aggregate_height=5, max_height=4, holes=1, bumpiness=3)
    config = RewardConfig()
    result = calculate_reward(before, after, 2, False, False, config)
    assert result.line_reward == config.line_rewards[2]
    assert result.holes_penalty == -0.75
    assert result.aggregate_height_penalty == -0.2
    assert result.bumpiness_penalty == -0.3
    assert result.terminal_penalty == 0.0
    assert result.truncation_penalty == 0.0
    assert result.total == pytest.approx(
        result.line_reward
        + result.holes_penalty
        + result.aggregate_height_penalty
        + result.bumpiness_penalty
    )


def test_reward_can_disable_shaping_and_separate_end_conditions() -> None:
    metrics = BoardMetrics((0,), aggregate_height=0, max_height=0, holes=0, bumpiness=0)
    config = RewardConfig(enable_shaping=False, terminal_penalty=-9.0, truncation_penalty=-2.0)
    terminated = calculate_reward(metrics, metrics, 0, True, False, config)
    truncated = calculate_reward(metrics, metrics, 0, False, True, config)
    assert terminated.total == -9.0
    assert truncated.total == -2.0
