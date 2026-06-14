import pytest

from speaktodeck import segment


def test_split_empty_returns_empty():
    assert segment.split_sentences("") == []
    assert segment.split_sentences("   ") == []


def test_clean_dedupes_and_trims():
    # _clean is exercised via split with a stub-free, deterministic backend.
    raw = ["  Hello.  ", "Hello.", "", "How are you?"]
    cleaned = segment._clean(raw)
    assert cleaned == ["Hello.", "How are you?"]


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        segment.split_sentences("Hello.", backend="nope")


def test_pysbd_backend_splits_punctuated_text():
    pytest.importorskip("pysbd")
    text = "I went to the store. Then I came home. It was raining!"
    sentences = segment.split_sentences(text, backend="pysbd")
    assert len(sentences) == 3
    assert sentences[0] == "I went to the store."
    assert sentences[-1] == "It was raining!"


@pytest.mark.network
def test_sat_backend_splits_unpunctuated_text():
    pytest.importorskip("wtpsplit")
    text = "i went to the store then i came home it was raining"
    sentences = segment.split_sentences(text, backend="sat")
    assert len(sentences) >= 2
