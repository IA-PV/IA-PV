from collections import deque
import json

import pytest

from tetris_ai.core.metrics import calculate_metrics
from tetris_ai.core.tetromino import PieceType
from tetris_ai.env import (
    Action,
    CANONICAL_MAX_PIECES,
    CANONICAL_RULESET_VERSION,
    EpisodeFinishedError,
    InvalidActionError,
    ScoringConfig,
    TetrisConfig,
    TetrisEnv,
)


def _set_state(
    env: TetrisEnv,
    *,
    current: PieceType,
    preview: tuple[PieceType, ...],
    board: tuple[tuple[int, ...], ...] | None = None,
    hold: PieceType | None = None,
    lines: int = 0,
) -> None:
    if board is not None:
        env._board._grid = [list(row) for row in board]
    env._current_piece = current
    env._preview = deque(preview)
    env._hold_piece = hold
    env._total_lines_cleared = lines
    env._terminated = False
    env._truncated = False
    env._terminal_reasons = ()
    env._truncation_reasons = ()
    env._last_metrics = calculate_metrics(env._board)
    env._invalidate_legal_actions()


def test_reset_is_seeded_and_observation_is_immutable() -> None:
    env = TetrisEnv(seed=7)
    first, first_info = env.reset(7)
    second, second_info = env.reset(7)
    assert first == second
    assert len(first.next_pieces) == env.config.preview_count
    assert first.max_pieces == CANONICAL_MAX_PIECES
    assert first.remaining_pieces == CANONICAL_MAX_PIECES
    assert first_info["config_id"] == second_info["config_id"]
    assert first_info["rng_reseeded"] is True
    assert first_info["horizon_mode"] == "finite"
    assert first_info["remaining_pieces"] == CANONICAL_MAX_PIECES
    with pytest.raises(TypeError):
        first.board[0][0] = 1  # type: ignore[index]


def test_reset_without_seed_continues_the_private_rng_stream() -> None:
    env = TetrisEnv(seed=7)
    _, info = env.reset()
    assert info["rng_reseeded"] is False
    assert info["seed"] == 7


def test_step_returns_gymnasium_style_result_and_rich_info() -> None:
    env = TetrisEnv(seed=4)
    action = next(action for action in env.legal_actions() if not action.is_hold)
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation.total_pieces_placed == 1
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["action"] == action.as_dict(env.width)
    assert info["placed_piece"] in {piece.value for piece in PieceType}
    assert "board_metrics_before" in info and "board_metrics_after" in info
    assert "reward" in info and "state_hash" in info
    json.dumps(info)


def test_fixed_action_encoding_and_mask_match_legal_actions() -> None:
    env = TetrisEnv(seed=4)
    observation = env.get_observation()
    assert len(observation.action_mask) == env.action_space_size == 8 * env.width
    legal_ids = {action.to_id(env.width) for action in env.legal_actions()}
    masked_ids = {index for index, enabled in enumerate(observation.action_mask) if enabled}
    assert legal_ids == masked_ids
    for action in env.legal_actions():
        assert Action.from_id(action.to_id(env.width), env.width) == action


def test_invalid_action_is_rejected_without_mutating_state() -> None:
    env = TetrisEnv(seed=1)
    before_hash = env.state_hash()
    with pytest.raises(InvalidActionError):
        env.step(Action(rotation=99, column=99))
    assert env.state_hash() == before_hash


def test_environment_clone_is_exact_and_independent() -> None:
    env = TetrisEnv(seed=8)
    clone = env.clone()
    action = next(action for action in env.legal_actions() if not action.is_hold)
    original_result = env.step(action)
    clone_result = clone.step(action)
    assert original_result == clone_result
    clone.step(next(action for action in clone.legal_actions() if not action.is_hold))
    assert env.total_pieces_placed == 1
    assert clone.total_pieces_placed == 2


def test_public_forward_model_never_refills_from_private_rng() -> None:
    env = TetrisEnv(seed=8)
    visible_queue = env.next_pieces
    context = env.decision_context()
    action = next(action for action in context.legal_actions if not action.is_hold)

    simulated = context.simulate(action)
    live_observation, _, _, _, _ = env.step(action)

    assert simulated.context.observation.current_piece == visible_queue[0]
    assert simulated.context.observation.next_pieces == visible_queue[1:]
    assert live_observation.next_pieces[:-1] == visible_queue[1:]
    assert len(live_observation.next_pieces) == env.config.preview_count


def test_hold_and_placement_are_one_atomic_transition() -> None:
    env = TetrisEnv(seed=3)
    original_current = env.current_piece
    original_queue = env.next_pieces
    action = next(action for action in env.legal_actions() if action.is_hold)
    preview = env.describe_action(action)

    observation, _, terminated, truncated, info = env.step(action)

    assert preview.piece == original_queue[0]
    assert observation.hold_piece == original_current
    assert observation.current_piece == original_queue[1]
    assert observation.total_pieces_placed == 1
    assert info["hold_used"] is True
    assert info["placed_piece"] == original_queue[0].value
    assert not terminated and not truncated


def test_score_uses_level_before_the_line_clear() -> None:
    env = TetrisEnv(seed=0)
    board = tuple([(0,) * 10] * 19 + [(1,) * 9 + (0,)])
    _set_state(
        env,
        current=PieceType.I,
        preview=(PieceType.O,) * env.config.preview_count,
        board=board,
        lines=9,
    )

    observation, reward, _, _, info = env.step(Action(rotation=1, column=9))

    assert info["lines_cleared"] == 1
    assert info["level_before"] == 1
    assert observation.level == 2
    assert info["score_gain"] == 100
    assert reward == 1.0
    assert info["reward"]["task_reward"] == 1.0  # type: ignore[index]


@pytest.mark.parametrize("enabled, expected_score", [(False, 0), (True, 400)])
def test_approximate_t_spin_scoring_is_explicitly_configurable(
    enabled: bool,
    expected_score: int,
) -> None:
    config = TetrisConfig(
        scoring=ScoringConfig(enable_approximate_t_spins=enabled),
    )
    env = TetrisEnv(config=config, seed=0)
    board = [list(row) for row in ((0,) * env.width,) * env.height]
    board[19][1] = 1
    _set_state(
        env,
        current=PieceType.T,
        preview=(PieceType.O,) * env.config.preview_count,
        board=tuple(tuple(row) for row in board),
    )

    _, _, _, _, info = env.step(Action(rotation=1, column=0))

    assert info["is_t_spin"] is enabled
    assert info["score_gain"] == expected_score


def test_back_to_back_survives_no_clear_and_breaks_on_regular_clear() -> None:
    env = TetrisEnv(seed=0)
    first_score, first_bonus = env._score_for_lock(lines_cleared=4, is_t_spin=False)
    zero_score, zero_bonus = env._score_for_lock(lines_cleared=0, is_t_spin=False)
    second_score, second_bonus = env._score_for_lock(lines_cleared=4, is_t_spin=False)
    regular_score, regular_bonus = env._score_for_lock(lines_cleared=1, is_t_spin=False)
    assert (first_score, first_bonus) == (800, False)
    assert (zero_score, zero_bonus) == (0, False)
    assert (second_score, second_bonus) == (1200, True)
    assert (regular_score, regular_bonus) == (100, False)
    assert env.back_to_back_active is False


def test_finite_horizon_is_a_true_terminal_with_no_penalty() -> None:
    env = TetrisEnv(max_pieces=1, seed=0)
    action = next(action for action in env.legal_actions() if not action.is_hold)
    observation, _, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    assert observation.done
    assert observation.remaining_pieces == 0
    assert not any(observation.action_mask)
    assert env.termination_reason == "horizon_completed"
    assert info["remaining_pieces"] == 0
    assert info["reward"]["terminal_penalty"] == 0.0  # type: ignore[index]
    assert info["reward"]["truncation_penalty"] == 0.0  # type: ignore[index]
    with pytest.raises(EpisodeFinishedError):
        env.step(action)


def test_external_time_limit_preserves_bootstrap_actions_but_prohibits_step() -> None:
    env = TetrisEnv(max_pieces=1, horizon_mode="time_limit", seed=0)
    action = next(action for action in env.legal_actions() if not action.is_hold)

    observation, _, terminated, truncated, info = env.step(action)

    assert not terminated and truncated
    assert env.termination_reason == "piece_limit"
    assert info["horizon_mode"] == "time_limit"
    assert env.legal_actions()
    legal_ids = {legal.to_id(env.width) for legal in env.legal_actions()}
    masked_ids = {index for index, enabled in enumerate(observation.action_mask) if enabled}
    assert legal_ids == masked_ids
    with pytest.raises(EpisodeFinishedError):
        env.step(env.legal_actions()[0])


def test_game_over_and_finite_horizon_can_coexist_on_the_same_step() -> None:
    config = TetrisConfig(width=5, height=4, max_pieces=1, preview_count=2, allow_hold=False)
    env = TetrisEnv(config=config, seed=0)
    board = (
        (0, 0, 1, 1, 0),
        (0, 0, 1, 1, 0),
        (1, 1, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    _set_state(env, current=PieceType.O, preview=(PieceType.O, PieceType.O), board=board)

    observation, _, terminated, truncated, info = env.step(Action(rotation=0, column=0))

    assert terminated and not truncated
    assert observation.terminated and not observation.truncated
    assert env.termination_reasons == ("game_over", "horizon_completed")
    assert info["reward"]["terminal_penalty"] == config.reward.terminal_penalty  # type: ignore[index]


def test_public_preview_exhaustion_remains_a_truncation() -> None:
    env = TetrisEnv(max_pieces=5, seed=0)
    _set_state(env, current=PieceType.O, preview=())
    action = next(action for action in env.legal_actions() if not action.is_hold)

    observation, _, terminated, truncated, info = env.step(action)

    assert not terminated and truncated
    assert observation.truncated
    assert env.termination_reason == "preview_horizon"
    assert not env.legal_actions()
    assert info["termination_reasons"] == ("preview_horizon",)


def test_hold_placement_can_rescue_an_otherwise_blocked_piece() -> None:
    config = TetrisConfig(width=5, height=4, preview_count=2, allow_hold=True)
    env = TetrisEnv(config=config, seed=0)
    board = (
        (0, 1, 0, 1, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 0, 1, 0),
    )
    _set_state(
        env,
        current=PieceType.O,
        preview=(PieceType.O, PieceType.O),
        hold=PieceType.I,
        board=board,
    )

    assert env.legal_actions()
    assert all(action.use_hold for action in env.legal_actions())
    _, _, terminated, _, info = env.step(Action.hold(rotation=1, column=0))
    assert info["placed_piece"] == PieceType.I.value
    assert terminated


def test_raw_observation_omits_engineered_metrics() -> None:
    env = TetrisEnv(config=TetrisConfig(observation_mode="raw"), seed=0)
    assert env.get_observation().metrics is None


def test_config_validation_and_fingerprint_are_stable() -> None:
    first = TetrisConfig()
    second = TetrisConfig()
    assert first.fingerprint == second.fingerprint
    assert first.max_pieces == CANONICAL_MAX_PIECES
    assert first.ruleset_version == CANONICAL_RULESET_VERSION
    assert first.horizon_mode == "finite"
    assert first.fingerprint != TetrisConfig(horizon_mode="time_limit").fingerprint
    with pytest.raises(ValueError):
        TetrisConfig(preview_count=0)
    with pytest.raises(ValueError):
        TetrisConfig(horizon_mode="invalid")  # type: ignore[arg-type]


def _naive_is_valid(board, shape, column, row) -> bool:
    """Reference collision check scanning the full shape bounding box."""

    for shape_row, cells in enumerate(shape):
        for shape_column, occupied in enumerate(cells):
            if not occupied:
                continue
            x, y = column + shape_column, row + shape_row
            if x < 0 or x >= board.width or y < 0 or y >= board.height:
                return False
            if board._grid[y][x]:
                return False
    return True


def _naive_drop_row(board, shape, column):
    if not _naive_is_valid(board, shape, column, 0):
        return None
    row = 0
    while _naive_is_valid(board, shape, column, row + 1):
        row += 1
    return row


def test_fast_placement_checks_match_naive_scan_on_arbitrary_boards() -> None:
    """The filled-cell fast path must be bit-for-bit identical to a full scan.

    Random boards include overhangs and floating cells, so this covers the
    notch-over-bump hard drops a surface-height shortcut would get wrong.
    """

    import random

    from tetris_ai.core.board import Board
    from tetris_ai.core.tetromino import rotations_for

    rng = random.Random(20260723)
    env = TetrisEnv(seed=0)
    for _ in range(120):
        board = Board(width=10, height=20)
        density = rng.choice((0.1, 0.25, 0.5, 0.75))
        board._grid = [
            [1 if rng.random() < density else 0 for _ in range(board.width)]
            for _ in range(board.height)
        ]
        env._board = board
        env._invalidate_legal_actions()
        for piece in PieceType:
            # 1. The optimized primitives match a naive full-bounding-box scan.
            for rotation, shape in enumerate(rotations_for(piece)):
                shape_width = len(shape[0])
                for column in range(board.width - shape_width + 1):
                    assert board.drop_row(shape, column) == _naive_drop_row(
                        board, shape, column
                    )
                    for row in range(-1, board.height + 1):
                        assert board.is_valid_position(
                            shape, column, row
                        ) == _naive_is_valid(board, shape, column, row)
            # 2. The O(cells) placement enumeration matches the row-by-row drop.
            fast = {
                (action.rotation, action.column): placement.row
                for action, placement in env._placements_for_piece(
                    piece, use_hold=False
                ).items()
            }
            reference: dict[tuple[int, int], int] = {}
            for rotation, shape in enumerate(rotations_for(piece)):
                shape_width = len(shape[0])
                for column in range(board.width - shape_width + 1):
                    row = _naive_drop_row(board, shape, column)
                    if row is not None:
                        reference[(rotation, column)] = row
            assert fast == reference
