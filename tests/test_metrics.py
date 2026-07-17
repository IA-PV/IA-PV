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
