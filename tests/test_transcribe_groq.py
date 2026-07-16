"""Groq transcription branch — mocked HTTP, no model loads."""

import requests

from speaktodeck import config, transcribe


class _FakeResp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"text": "  Hello   from Groq.  "}


def test_groq_branch_posts_file_and_normalizes(monkeypatch, tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"FAKEAUDIO")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs, url=url)
        return _FakeResp()

    monkeypatch.setattr(requests, "post", fake_post)

    text = transcribe.transcribe_audio(str(audio))

    assert text == "Hello from Groq."
    assert captured["url"] == config.GROQ_TRANSCRIBE_URL
    assert captured["headers"]["Authorization"] == "Bearer gsk_test"
    assert captured["data"]["model"] == config.GROQ_WHISPER_MODEL
    assert captured["files"]["file"][0] == "clip.mp3"


def test_no_key_means_local_path(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert transcribe.groq_is_enabled() is False
    # preload with a key set must be a no-op (would OOM small hosts otherwise)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    transcribe.preload()  # would try to load a model and fail if not guarded
