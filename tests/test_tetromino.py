from tetris_ai.core.tetromino import PieceType, rotations_for


def test_tetrominoes_have_the_expected_unique_rotation_counts() -> None:
    expected_counts = {
        PieceType.O: 1,
        PieceType.I: 2,
        PieceType.S: 2,
        PieceType.Z: 2,
        PieceType.T: 4,
        PieceType.J: 4,
        PieceType.L: 4,
    }
    assert {piece: len(rotations_for(piece)) for piece in PieceType} == expected_counts
