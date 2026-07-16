from tetris_ai.core.tetromino import PieceType
from tetris_ai.env.action import Action
from tetris_ai.visualization.tk_viewer import VisualBoard


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
