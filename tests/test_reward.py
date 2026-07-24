import pytest

from tetris_ai.core.metrics import BoardMetrics
from tetris_ai.env import (
    CANONICAL_LINE_REWARDS,
    RewardConfig,
    canonical_reward_config,
    rl_training_reward_config,
)
from tetris_ai.env.reward import calculate_reward


def test_canonical_reward_is_clean_and_stationary() -> None:
    config = canonical_reward_config()
    assert config == RewardConfig()
    assert config.line_rewards == CANONICAL_LINE_REWARDS
    assert config.enable_shaping is False
    assert config.terminal_penalty == 0.0
    assert config.truncation_penalty == 0.0


def test_reward_breakdown_uses_potential_differences() -> None:
    before = BoardMetrics((2, 1), aggregate_height=3, max_height=2, holes=0, bumpiness=1)
    after = BoardMetrics((4, 1), aggregate_height=5, max_height=4, holes=1, bumpiness=3)
    config = rl_training_reward_config()
    result = calculate_reward(before, after, 2, False, False, config)
    assert result.line_reward == config.line_rewards[2]
    assert result.holes_penalty == -0.5
    assert result.aggregate_height_penalty == pytest.approx(-0.1)
    assert result.bumpiness_penalty == pytest.approx(-0.2)
    assert result.terminal_penalty == 0.0
    assert result.truncation_penalty == 0.0
    assert result.task_reward == result.line_reward == 3.0
    assert result.potential_shaping == pytest.approx(-0.8)
    assert result.total == pytest.approx(result.task_reward + result.potential_shaping)


def test_reward_can_disable_shaping_and_separate_end_conditions() -> None:
    metrics = BoardMetrics((0,), aggregate_height=0, max_height=0, holes=0, bumpiness=0)
    config = RewardConfig(enable_shaping=False, terminal_penalty=-9.0, truncation_penalty=-2.0)
    terminated = calculate_reward(metrics, metrics, 0, True, False, config)
    truncated = calculate_reward(metrics, metrics, 0, False, True, config)
    assert terminated.total == -9.0
    assert truncated.total == -2.0
    assert terminated.task_reward == -9.0
    assert terminated.potential_shaping == 0.0


def test_true_terminal_zeroes_successor_potential_but_external_truncation_does_not() -> None:
    before = BoardMetrics((2,), aggregate_height=2, max_height=2, holes=2, bumpiness=0)
    after = BoardMetrics((3,), aggregate_height=3, max_height=3, holes=3, bumpiness=0)
    config = RewardConfig(
        holes_delta_weight=-0.5,
        enable_shaping=True,
        shaping_gamma=0.9,
        shaping_scale=2.0,
    )

    terminal = calculate_reward(before, after, 0, True, False, config)
    time_limit = calculate_reward(before, after, 0, False, True, config)

    assert terminal.holes_penalty == pytest.approx(2.0)
    assert time_limit.holes_penalty == pytest.approx(-0.7)


def test_pbrs_telescopes_to_zero_from_empty_board_through_terminal() -> None:
    initial = BoardMetrics((0,), aggregate_height=0, max_height=0, holes=0, bumpiness=0)
    middle = BoardMetrics((2,), aggregate_height=2, max_height=2, holes=1, bumpiness=3)
    terminal_board = BoardMetrics(
        (4,), aggregate_height=4, max_height=4, holes=2, bumpiness=1
    )
    config = rl_training_reward_config()

    first = calculate_reward(initial, middle, 0, False, False, config)
    last = calculate_reward(middle, terminal_board, 0, True, False, config)

    assert first.potential_shaping + last.potential_shaping == pytest.approx(0.0)
    assert first.total + last.total == pytest.approx(
        first.task_reward + last.task_reward
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"shaping_gamma": -0.01},
        {"shaping_gamma": 1.01},
        {"shaping_gamma": float("nan")},
        {"shaping_scale": -0.01},
        {"shaping_scale": float("inf")},
        {"terminal_penalty": float("-inf")},
        {"line_rewards": (0.0, 1.0, 3.0, 5.0, float("nan"))},
    ],
)
def test_reward_config_rejects_invalid_numeric_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RewardConfig(**overrides)  # type: ignore[arg-type]
