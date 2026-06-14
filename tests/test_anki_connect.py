import json

import pytest

from speaktodeck import anki_connect
from speaktodeck.flashcards import build_candidates


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class Recorder:
    """Captures AnkiConnect calls and returns canned results in order."""

    def __init__(self, results=None, error=None):
        self.calls = []
        self.error = error
        self.results = results or {}

    def __call__(self, url, json=None, timeout=None):  # noqa: A002 - mirrors requests.post
        self.calls.append(json)
        if self.error:
            return FakeResponse({"result": None, "error": self.error})
        action = json["action"]
        return FakeResponse({"result": self.results.get(action), "error": None})


def test_push_cards_sends_createdeck_and_addnote(monkeypatch):
    rec = Recorder(results={"createDeck": 1, "addNote": 12345})
    monkeypatch.setattr("requests.post", rec, raising=False)
    import requests

    monkeypatch.setattr(requests, "post", rec)

    cards = build_candidates(["Hello."], ["Hola."])
    cards[0].status = "accepted"

    added = anki_connect.push_cards(cards, "MyDeck")
    assert added == 1

    actions = [c["action"] for c in rec.calls]
    assert actions == ["createDeck", "addNote"]

    create = rec.calls[0]
    assert create["params"]["deck"] == "MyDeck"
    assert create["version"] == 6

    note = rec.calls[1]["params"]["note"]
    assert note["deckName"] == "MyDeck"
    assert note["fields"]["Front"] == "Hello."
    assert note["fields"]["Back"] == "Hola."
    assert "speaktodeck" in note["tags"]


def test_push_cards_with_audio_stores_media_and_embeds_sound(monkeypatch):
    rec = Recorder(results={"createDeck": 1, "storeMediaFile": "f.mp3", "addNote": 1})
    import requests

    monkeypatch.setattr(requests, "post", rec)

    cards = build_candidates(["Hi."], ["Hola."], audio=[b"AUDIO"])
    cards[0].status = "accepted"
    anki_connect.push_cards(cards, "D")

    actions = [c["action"] for c in rec.calls]
    assert actions == ["createDeck", "storeMediaFile", "addNote"]

    media = rec.calls[1]["params"]
    assert media["data"]  # base64 string present
    note_back = rec.calls[2]["params"]["note"]["fields"]["Back"]
    assert "[sound:" in note_back


def test_anki_error_raises(monkeypatch):
    rec = Recorder(error="deck not found")
    import requests

    monkeypatch.setattr(requests, "post", rec)

    cards = build_candidates(["Hi."], ["Hola."])
    cards[0].status = "accepted"
    with pytest.raises(anki_connect.AnkiConnectError):
        anki_connect.push_cards(cards, "D")


def test_is_available_false_on_connection_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    import requests

    monkeypatch.setattr(requests, "post", boom)
    assert anki_connect.is_available() is False
