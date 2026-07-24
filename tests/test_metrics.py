from tetris_ai.core.board import Board
from tetris_ai.core.metrics import calculate_metrics


def test_board_metrics() -> None:
    board = Board(width=3, height=4)
    board.place(((1,),), 0, 1)
    board.place(((1,),), 0, 3)
    board.place(((1,),), 1, 3)
    metrics = calculate_metrics(board, lines_cleared_last_move=2)
    assert metrics.column_heights == (3, 1, 0)
    assert metrics.aggregate_height == 4
    assert metrics.max_height == 3
    assert metrics.holes == 1
    assert metrics.bumpiness == 3
    assert metrics.lines_cleared_last_move == 2
    assert metrics.row_transitions == 8
    assert metrics.column_transitions == 5
    assert metrics.wells == 1


def test_empty_board_surface_metrics_have_no_wells_or_row_transitions() -> None:
    metrics = calculate_metrics(Board(width=3, height=4))

    assert metrics.wells == 0
    # A flat empty surface still flips at each side wall on every row.
    assert metrics.row_transitions == 8
    # Every column flips once (empty column -> filled floor).
    assert metrics.column_transitions == 3


def test_full_rows_contribute_no_row_transitions() -> None:
    board = Board(width=3, height=2)
    for column in range(3):
        board.place(((1,),), column, 1)

    metrics = calculate_metrics(board)

    # The completed bottom row has zero horizontal flips; the empty top row
    # flips twice (once at each wall).
    assert metrics.row_transitions == 2


def test_deep_well_is_counted_by_depth() -> None:
    board = Board(width=3, height=4)
    # Fill the two outer columns to height 3, leaving the middle as a depth-3 well.
    for row in range(1, 4):
        board.place(((1,),), 0, row)
        board.place(((1,),), 2, row)

    metrics = calculate_metrics(board)

    assert metrics.column_heights == (3, 0, 3)
    assert metrics.wells == 3
