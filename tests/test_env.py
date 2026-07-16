import pytest

from tetris_ai.core.tetromino import PieceType
from tetris_ai.env import Action, InvalidActionError, TetrisEnv


def test_reset_is_seeded_and_observation_is_immutable() -> None:
    env = TetrisEnv(seed=7)
    first = env.reset(7)
    second = env.reset(7)
    assert first.current_piece == second.current_piece
    with pytest.raises(TypeError):
        first.board[0][0] = 1  # type: ignore[index]


def test_step_returns_turn_result_and_legal_actions_are_valid() -> None:
    env = TetrisEnv(seed=4)
    action = next(action for action in env.legal_actions() if not action.is_hold)
    observation, reward, done, info = env.step(action)
    assert observation.total_pieces_placed == 1
    assert isinstance(reward, float)
    assert done is False
    assert info["action"] == action
    assert "reward" in info and "termination_reason" in info


def test_invalid_action_is_rejected() -> None:
    env = TetrisEnv(seed=1)
    with pytest.raises(InvalidActionError):
        env.step(Action(rotation=99, column=99))


def test_environment_clone_is_independent() -> None:
    env = TetrisEnv(seed=8)
    clone = env.clone()
    clone.step(next(action for action in clone.legal_actions() if not action.is_hold))
    assert env.total_pieces_placed == 0
    assert clone.total_pieces_placed == 1


def test_hold_action_swaps_current_piece_without_placing() -> None:
    env = TetrisEnv(seed=3)
    original_current = env.current_piece
    original_next = env.next_piece
    observation, reward, done, info = env.step(Action.hold())
    assert observation.hold_piece == original_current
    assert observation.current_piece == original_next
    assert observation.total_pieces_placed == 0
    assert observation.can_hold is False
    assert reward == 0.0
    assert done is False
    assert info["hold_used"] is True
    assert Action.hold() not in env.legal_actions()


def test_level_and_score_scale_after_line_clear() -> None:
    env = TetrisEnv(seed=0)
    env.current_piece = PieceType.I
    env.next_piece = PieceType.O
    env.total_lines_cleared = 9
    env.board._grid[-1] = [1] * 9 + [0]
    action = Action(rotation=1, column=9)
    observation, _, _, info = env.step(action)
    assert info["lines_cleared"] == 1
    assert observation.level == 2
    assert info["score_gain"] == 200
    assert observation.score == 200


def test_t_spin_is_detected_when_three_center_corners_are_filled() -> None:
    env = TetrisEnv(seed=0)
    env.current_piece = PieceType.T
    env.next_piece = PieceType.O
    env.board._grid[19][1] = 1
    _, _, _, info = env.step(Action(rotation=1, column=0))
    assert info["is_t_spin"] is True
    assert info["score_gain"] == 400


def test_back_to_back_bonus_applies_to_consecutive_special_clears() -> None:
    env = TetrisEnv(seed=0)
    first_score, first_bonus = env._score_for_lock(lines_cleared=4, is_t_spin=False)
    second_score, second_bonus = env._score_for_lock(lines_cleared=4, is_t_spin=False)
    regular_score, regular_bonus = env._score_for_lock(lines_cleared=1, is_t_spin=False)
    assert first_score == 800
    assert first_bonus is False
    assert second_score == 1200
    assert second_bonus is True
    assert regular_score == 100
    assert regular_bonus is False
    assert env.back_to_back_active is False
