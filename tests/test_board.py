from tetris_ai.core.board import Board


DOT = ((1,),)


def test_collision_with_walls_and_floor() -> None:
    board = Board(width=4, height=4)
    assert not board.is_valid_position(DOT, -1, 0)
    assert not board.is_valid_position(DOT, 4, 0)
    assert not board.is_valid_position(DOT, 0, 4)
    assert board.drop_row(DOT, 2) == 3


def test_clear_complete_line() -> None:
    board = Board(width=4, height=4)
    board.place(((1, 1, 1, 1),), 0, 3)
    assert board.clear_lines() == 1
    assert board.matrix() == ((0, 0, 0, 0),) * 4


def test_clone_is_independent() -> None:
    board = Board(width=2, height=2)
    clone = board.clone()
    clone.place(DOT, 0, 1)
    assert board.matrix() != clone.matrix()
