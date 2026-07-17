from tetris_ai.core.randomizer import SevenBagRandomizer
from tetris_ai.core.tetromino import PieceType


def test_seven_bag_seed_is_repeatable_and_contains_all_pieces() -> None:
    first = SevenBagRandomizer(123)
    second = SevenBagRandomizer(123)
    first_sequence = [first.next_piece() for _ in range(14)]
    assert first_sequence == [second.next_piece() for _ in range(14)]
    assert set(first_sequence[:7]) == set(PieceType)
    assert set(first_sequence[7:]) == set(PieceType)
