"""Push flashcards straight into the running Anki desktop app via AnkiConnect.

Requires the AnkiConnect add-on (code 2055492159) installed and Anki open.
The add-on serves an HTTP API on http://localhost:8765.
"""

from __future__ import annotations

import base64

from . import config
from .flashcards import Candidate


class AnkiConnectError(RuntimeError):
    """Raised when AnkiConnect is unreachable or returns an error."""


def _invoke(action: str, **params):
    """POST a single AnkiConnect action and return its ``result``."""
    import requests

    payload = {
        "action": action,
        "version": config.ANKI_CONNECT_VERSION,
        "params": params,
    }
    try:
        resp = requests.post(config.ANKI_CONNECT_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # network error, Anki closed, bad JSON, ...
        raise AnkiConnectError(
            "Could not reach Anki. Make sure Anki is open and the AnkiConnect "
            "add-on (2055492159) is installed."
        ) from exc

    if data.get("error") is not None:
        raise AnkiConnectError(str(data["error"]))
    return data.get("result")


# AnkiConnect returns this when the HTTP server is up but no collection is
# loaded — Anki is open on the profile picker, mid-startup, or a modal dialog
# (sync/import/add-on window) is blocking access to the collection.
_NO_COLLECTION_MARKER = "collection is not available"


def status() -> tuple[bool, str]:
    """Probe AnkiConnect and classify the result for the UI.

    Returns ``(connected, message)``. ``connected`` is True only when a
    collection is loaded and decks are queryable. When False, ``message`` is a
    markdown string explaining the specific reason so the sidebar can guide the
    user instead of lumping every failure into a generic "not detected".
    """
    try:
        _invoke("deckNames")
        return True, "Anki: connected"
    except AnkiConnectError as exc:
        if _NO_COLLECTION_MARKER in str(exc).lower():
            return False, (
                "**Anki: no profile loaded**\n\nAnki is open but its collection "
                "isn't available. Click into your profile (and close any open "
                "dialog) so the deck list is showing, then rerun."
            )
        return False, (
            "**Anki: not detected**\n\nOpen Anki with the AnkiConnect add-on "
            "(2055492159) installed."
        )


def _media_filename(card: Candidate) -> str:
    return f"speaktodeck_{card.id}.{card.audio_ext}"


def push_cards(cards: list[Candidate], deck_name: str | None = None) -> int:
    """Create ``deck_name`` (if needed) and add a note per card.

    Audio, when present, is stored in Anki's media collection and embedded on
    the Back field as ``[sound:...]``. Returns the number of notes added.
    Raises ``AnkiConnectError`` on failure so callers can offer the .apkg
    fallback.
    """
    deck_name = deck_name or config.DEFAULT_DECK_NAME
    _invoke("createDeck", deck=deck_name)

    added = 0
    for card in cards:
        back = card.translation
        if card.audio:
            filename = _media_filename(card)
            _invoke(
                "storeMediaFile",
                filename=filename,
                data=base64.b64encode(card.audio).decode("ascii"),
            )
            back = f"{card.translation}<br>[sound:{filename}]"

        _invoke(
            "addNote",
            note={
                "deckName": deck_name,
                "modelName": config.ANKI_NOTE_MODEL,
                "fields": {"Front": card.english, "Back": back},
                "options": {
                    "allowDuplicate": False,
                    "duplicateScope": "deck",
                },
                "tags": ["speaktodeck"],
            },
        )
        added += 1

    return added
