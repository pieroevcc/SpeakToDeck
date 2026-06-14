import io
import zipfile

import pytest

from speaktodeck.flashcards import build_candidates

genanki = pytest.importorskip("genanki")
from speaktodeck import anki_export  # noqa: E402  (after importorskip)


def _accepted(english, translations, audio=None):
    cards = build_candidates(english, translations, audio=audio)
    for c in cards:
        c.status = "accepted"
    return cards


def test_export_returns_valid_apkg_bytes():
    cards = _accepted(["Hello.", "Bye."], ["Hola.", "Adiós."])
    data = anki_export.export_deck(cards, "TestDeck")
    assert isinstance(data, bytes) and len(data) > 0
    # .apkg is a zip archive
    assert data[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
    # genanki packages contain the sqlite collection
    assert any("collection.anki2" in n for n in names)


def test_export_with_audio_includes_media():
    cards = _accepted(["Hi."], ["Hola."], audio=[b"FAKEAUDIO"])
    data = anki_export.export_deck(cards, "AudioDeck")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
    # genanki stores media under numeric names plus a "media" manifest
    assert "media" in names


def test_deck_id_is_stable():
    assert anki_export._deck_id("SpeakToDeck") == anki_export._deck_id("SpeakToDeck")
