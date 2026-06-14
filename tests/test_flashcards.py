import pytest

from speaktodeck import flashcards


def test_build_candidates_zips_and_defaults():
    cards = flashcards.build_candidates(
        english=["Hello.", "How are you?"],
        translations=["Hola.", "¿Cómo estás?"],
    )
    assert len(cards) == 2
    assert cards[0].english == "Hello." and cards[0].translation == "Hola."
    assert all(c.status == "pending" for c in cards)
    assert all(c.audio is None for c in cards)
    # ids are unique
    assert len({c.id for c in cards}) == 2


def test_build_candidates_with_audio_and_flags():
    cards = flashcards.build_candidates(
        english=["A.", "B."],
        translations=["a.", "b."],
        audio=[b"\x00", None],
        translation_ok=[True, False],
    )
    assert cards[0].audio == b"\x00"
    assert cards[1].audio is None
    assert cards[1].translation_ok is False


def test_build_candidates_length_mismatch_raises():
    with pytest.raises(ValueError):
        flashcards.build_candidates(english=["one"], translations=["a", "b"])


def test_accepted_filters_status():
    cards = flashcards.build_candidates(["A.", "B.", "C."], ["a", "b", "c"])
    cards[0].status = "accepted"
    cards[1].status = "discarded"
    cards[2].status = "accepted"
    acc = flashcards.accepted(cards)
    assert [c.english for c in acc] == ["A.", "C."]
