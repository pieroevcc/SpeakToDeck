"""Pronunciation audio for translated sentences.

Two engines, tried in order:

* **NVIDIA Magpie multilingual TTS** (primary) — NVIDIA's hosted neural TTS,
  reached over Riva gRPC and authenticated with ``NVIDIA_API_KEY``. High quality
  for the languages NVIDIA covers (see ``config.NVIDIA_TTS_VOICES``).
* **edge-tts** (fallback) — free, key-less Microsoft Edge neural voices that
  cover every target language. Used whenever NVIDIA can't voice a language, no
  key is set, or the NVIDIA call fails for any reason.

The public surface is ``synthesize(text, lang) -> tuple[bytes, str] | None``,
returning ``(audio_bytes, extension)`` where extension is ``"wav"`` (NVIDIA) or
``"mp3"`` (edge). It returns ``None`` only when neither engine can voice the
text, so a missing voice degrades to a card without audio rather than breaking
the run.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor

from . import config

log = logging.getLogger(__name__)

# Cap on concurrent TTS calls. Synthesis is network-bound (NVIDIA gRPC / the
# edge-tts websocket), so threads overlap latency; each worker uses its own
# engine connection and asyncio loop, so they don't interfere.
_MAX_WORKERS = 6


def is_enabled() -> bool:
    """True if pronunciation audio can be produced (edge-tts needs no key)."""
    return True


def nvidia_is_enabled() -> bool:
    """True when an NVIDIA API key is set, so the NVIDIA TTS engine can run."""
    return bool(os.environ.get("NVIDIA_API_KEY"))


# --- NVIDIA Magpie TTS (primary) ---------------------------------------------


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container (no encoder needed)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit samples
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


def _synthesize_nvidia(text: str, lang: str) -> tuple[bytes, str] | None:
    """Synthesize via NVIDIA Magpie TTS. Returns ``(wav_bytes, "wav")`` or None.

    Returns None (so the caller falls back to edge-tts) when no key is set, the
    language isn't in NVIDIA's supported set, or the gRPC call fails.
    """
    if not nvidia_is_enabled():
        return None
    voice = config.NVIDIA_TTS_VOICES.get(lang)
    if not voice:
        return None  # language NVIDIA doesn't cover — let edge-tts handle it
    voice_name, language_code = voice

    try:
        import riva.client

        auth = riva.client.Auth(
            None,  # ssl_cert: use the public CA bundle
            True,  # use_ssl
            config.NVIDIA_TTS_SERVER,
            [
                ["function-id", config.NVIDIA_TTS_FUNCTION_ID],
                ["authorization", f"Bearer {os.environ['NVIDIA_API_KEY']}"],
            ],
        )
        service = riva.client.SpeechSynthesisService(auth)
        resp = service.synthesize(
            text,
            voice_name=voice_name,
            language_code=language_code,
            sample_rate_hz=config.NVIDIA_TTS_SAMPLE_RATE,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
        )
        if resp.audio:
            return _pcm_to_wav(resp.audio, config.NVIDIA_TTS_SAMPLE_RATE), "wav"
    except Exception:  # noqa: BLE001 - degrade to the edge-tts fallback
        log.warning(
            "NVIDIA TTS failed for %s; falling back to edge-tts", lang, exc_info=True
        )
    return None


# --- edge-tts (fallback) ------------------------------------------------------


async def _stream_mp3(text: str, voice: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    return bytes(buf)


def _synthesize_edge(text: str, lang: str) -> tuple[bytes, str] | None:
    """Synthesize via edge-tts. Returns ``(mp3_bytes, "mp3")`` or None."""
    voice = config.voice_for(lang)
    if not voice:
        return None

    # The Edge endpoint occasionally drops the websocket mid-handshake
    # (Windows: WinError 64). A couple of quick retries clears it.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            audio = asyncio.run(_stream_mp3(text, voice))
            if audio:
                return audio, "mp3"
        except Exception as exc:  # noqa: BLE001 - degrade to no-audio
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))

    log.warning("edge-tts synthesis failed for %s: %s", lang, last_exc)
    return None


# --- public surface -----------------------------------------------------------


def synthesize(text: str, lang: str) -> tuple[bytes, str] | None:
    """Return ``(audio_bytes, extension)`` for ``text`` in ``lang``, or None.

    Tries NVIDIA Magpie TTS first, then falls back to edge-tts. Returns None
    only when neither engine can voice the language or the text is empty.
    """
    text = (text or "").strip()
    if not text:
        return None
    return _synthesize_nvidia(text, lang) or _synthesize_edge(text, lang)


def synthesize_many(
    texts: list[str], lang: str
) -> list[tuple[bytes, str] | None]:
    """Synthesize a batch of texts concurrently, preserving input order.

    Same per-item contract as :func:`synthesize` (each entry is
    ``(audio_bytes, extension)`` or ``None``), but the network calls overlap so
    a multi-sentence card set isn't voiced one slow round-trip at a time.
    """
    if not texts:
        return []

    workers = min(_MAX_WORKERS, len(texts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda t: synthesize(t, lang), texts))
