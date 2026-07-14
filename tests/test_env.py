import pytest

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
    action = env.legal_actions()[0]
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
    clone.step(clone.legal_actions()[0])
    assert env.total_pieces_placed == 0
    assert clone.total_pieces_placed == 1
