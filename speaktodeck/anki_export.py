"""Fallback: export accepted cards as a downloadable .apkg deck (genanki).

Used when Anki / AnkiConnect isn't running. Audio, when present, is embedded
as a media file inside the package and referenced from the Back field.
"""

from __future__ import annotations

import os
import tempfile
import zlib

import genanki

from .flashcards import Candidate

# Stable, app-specific model id (random but fixed so re-imports update cleanly).
_MODEL_ID = 1607392319
_MODEL = genanki.Model(
    _MODEL_ID,
    "SpeakToDeck Basic",
    fields=[{"name": "English"}, {"name": "Translation"}],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "{{English}}",
            "afmt": '{{FrontSide}}<hr id="answer">{{Translation}}',
        }
    ],
)


def _deck_id(deck_name: str) -> int:
    """Derive a stable 31-bit deck id from the name."""
    return zlib.crc32(deck_name.encode("utf-8")) & 0x7FFFFFFF


def export_deck(cards: list[Candidate], deck_name: str = "SpeakToDeck") -> bytes:
    """Build an .apkg for the given cards and return its bytes."""
    deck = genanki.Deck(_deck_id(deck_name), deck_name)
    media_files: list[str] = []
    tmpdir = tempfile.mkdtemp(prefix="speaktodeck_")

    try:
        for card in cards:
            back = card.translation
            if card.audio:
                filename = f"speaktodeck_{card.id}.{card.audio_ext}"
                media_path = os.path.join(tmpdir, filename)
                with open(media_path, "wb") as fh:
                    fh.write(card.audio)
                media_files.append(media_path)
                back = f"{card.translation}<br>[sound:{filename}]"

            deck.add_note(
                genanki.Note(model=_MODEL, fields=[card.english, back], tags=["speaktodeck"])
            )

        package = genanki.Package(deck)
        package.media_files = media_files

        out_path = os.path.join(tmpdir, "deck.apkg")
        package.write_to_file(out_path)
        with open(out_path, "rb") as fh:
            return fh.read()
    finally:
        # Best-effort cleanup of the temp working dir.
        for root, _dirs, files in os.walk(tmpdir, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except OSError:
                    pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
