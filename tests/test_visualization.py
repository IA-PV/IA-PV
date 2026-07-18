import pytest

from tetris_ai.core.tetromino import PieceType
from tetris_ai.env.action import Action
from tetris_ai.visualization.tk_viewer import AnimationTiming, VisualBoard


def test_animation_timing_accelerates_by_level_and_respects_the_minimum() -> None:
    timing = AnimationTiming(base_delay_ms=80, min_delay_ms=18, level_speed_factor=0.85)

    assert timing.delay_for_level(1) == 80
    assert timing.delay_for_level(5) == 42
    assert timing.delay_for_level(10) == 19
    assert timing.delay_for_level(20) == 18


@pytest.mark.parametrize(
    ("base_delay_ms", "min_delay_ms", "level_speed_factor"),
    ((0, 18, 0.85), (80, 0, 0.85), (18, 80, 0.85), (80, 18, 0.0), (80, 18, 1.1)),
)
def test_animation_timing_rejects_invalid_configuration(
    base_delay_ms: int,
    min_delay_ms: int,
    level_speed_factor: float,
) -> None:
    with pytest.raises(ValueError):
        AnimationTiming(base_delay_ms, min_delay_ms, level_speed_factor)


def test_animation_timing_rejects_non_positive_level() -> None:
    with pytest.raises(ValueError):
        AnimationTiming().delay_for_level(0)


def test_visual_board_places_piece_and_clears_full_line() -> None:
    board = VisualBoard(width=4, height=4)
    board.place(PieceType.I, Action(rotation=0, column=0), row=3)
    removed = board.clear_full_lines()
    assert removed == 1
    assert all(cell is None for row in board.cells for cell in row)


def test_visual_board_syncs_empty_cells_from_environment_matrix() -> None:
    board = VisualBoard(width=2, height=2)
    board.place(PieceType.O, Action(rotation=0, column=0), row=0)
    board.sync_with_matrix(((1, 0), (0, 1)))
    assert board.cells[0][0] == PieceType.O
    assert board.cells[0][1] is None
    assert board.cells[1][0] is None
    assert board.cells[1][1] == PieceType.O
