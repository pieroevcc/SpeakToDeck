"""Offline tests for the TTS engine-selection logic.

These don't hit the network: the two engine helpers are monkeypatched so we only
exercise the NVIDIA-primary / edge-tts-fallback dispatch in ``synthesize``.
"""

from speaktodeck import tts


def test_empty_text_returns_none():
    assert tts.synthesize("   ", "es") is None


def test_prefers_nvidia_when_it_succeeds(monkeypatch):
    monkeypatch.setattr(tts, "_synthesize_nvidia", lambda t, l: (b"WAV", "wav"))
    # edge should never be consulted when NVIDIA succeeds
    monkeypatch.setattr(
        tts, "_synthesize_edge", lambda t, l: (_ for _ in ()).throw(AssertionError)
    )
    assert tts.synthesize("hello", "es") == (b"WAV", "wav")


def test_falls_back_to_edge_when_nvidia_returns_none(monkeypatch):
    monkeypatch.setattr(tts, "_synthesize_nvidia", lambda t, l: None)
    monkeypatch.setattr(tts, "_synthesize_edge", lambda t, l: (b"MP3", "mp3"))
    assert tts.synthesize("hello", "ko") == (b"MP3", "mp3")


def test_returns_none_when_both_engines_fail(monkeypatch):
    monkeypatch.setattr(tts, "_synthesize_nvidia", lambda t, l: None)
    monkeypatch.setattr(tts, "_synthesize_edge", lambda t, l: None)
    assert tts.synthesize("hello", "xx") is None


def test_nvidia_skipped_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert tts.nvidia_is_enabled() is False
    # _synthesize_nvidia bails out (returns None) before importing riva.client
    assert tts._synthesize_nvidia("hello", "es") is None


def test_pcm_to_wav_is_riff_container():
    wav = tts._pcm_to_wav(b"\x00\x01" * 8, 44100)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
