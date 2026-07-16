"""Local speech-to-text using faster-whisper (offline, no API key).

The model is loaded lazily and cached at module level so it is only
constructed once per process. Weights are downloaded once into a project-local
``models/`` directory as plain files (see ``_resolve_model``) and reused after.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache

from . import config

# Project-local model store. We deliberately avoid the default Hugging Face
# blob cache: on Windows it links snapshot files to blobs with os.symlink,
# which needs Developer Mode or admin rights (otherwise "[WinError 1314] A
# required privilege is not held by the client"). Downloading into a plain
# directory with real files via download_model(output_dir=...) sidesteps
# symlinks entirely, so it works for any user on any platform.
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _resolve_model(model: str) -> str:
    """Return a local directory path for ``model``.

    A path that already exists is used as-is; otherwise ``model`` is treated as
    a faster-whisper model name (or HF repo id) and materialized into
    ``_MODELS_DIR`` as real files. Re-runs are cheap: download_model verifies
    existing files against the hub and skips anything already present.
    """
    if os.path.isdir(model):
        return model

    from faster_whisper import download_model

    target = os.path.join(_MODELS_DIR, model.replace("/", "__"))
    # Already downloaded? Load straight from disk (no network, works offline).
    if os.path.isfile(os.path.join(target, "model.bin")):
        return download_model(model, output_dir=target, local_files_only=True)
    return download_model(model, output_dir=target)


@lru_cache(maxsize=3)
def _get_model(model: str, device: str, compute_type: str):
    # Imported lazily so the (heavy) dependency isn't required just to import
    # this module in tests that mock transcription. Cached per (model, device,
    # compute_type) so switching settings in the UI loads a fresh model and
    # keeps a few warm rather than reloading on every run.
    from faster_whisper import WhisperModel

    return WhisperModel(_resolve_model(model), device=device, compute_type=compute_type)


def groq_is_enabled() -> bool:
    """True when a Groq API key is set, so transcription is remote (no local model)."""
    return bool(os.environ.get("GROQ_API_KEY"))


class TranscriptionLimitError(RuntimeError):
    """Raised when the daily Groq transcription budget is spent."""


# Daily cap on Groq calls so a public demo can't burn through the key.
# ponytail: in-memory counter — resets on process restart; move to a store if
# the demo ever needs a hard guarantee.
_groq_calls = {"day": None, "n": 0}


def _check_groq_budget() -> None:
    limit = int(os.environ.get("GROQ_DAILY_LIMIT", "5"))
    today = date.today()
    if _groq_calls["day"] != today:
        _groq_calls.update(day=today, n=0)
    if _groq_calls["n"] >= limit:
        raise TranscriptionLimitError(
            f"Daily transcription limit reached ({limit}/day on this demo). "
            "Paste text instead — it skips transcription — or run the app locally."
        )
    _groq_calls["n"] += 1


def _transcribe_groq(path: str) -> str:
    """Transcribe via Groq's hosted Whisper (OpenAI-compatible endpoint).

    Raises on failure — with a key set the user asked for the remote path, and
    a silent local fallback would load a multi-GB model on hosts that can't.
    """
    import requests

    _check_groq_budget()
    with open(path, "rb") as fh:
        resp = requests.post(
            config.GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            files={"file": (os.path.basename(path), fh)},
            data={
                "model": config.GROQ_WHISPER_MODEL,
                "language": "en",
                "prompt": config.WHISPER_INITIAL_PROMPT,
            },
            timeout=120,
        )
    resp.raise_for_status()
    return " ".join(resp.json().get("text", "").split())


def preload(
    *,
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> None:
    """Construct and cache the Whisper model ahead of time.

    Called at app startup (in a background thread) so the first real
    transcription isn't paying the model-load cost. Safe to call repeatedly —
    ``_get_model`` is ``lru_cache``-d, so subsequent calls are no-ops.
    No-op when Groq handles transcription (loading a local model would OOM
    small hosts and waste RAM everywhere else).
    """
    if groq_is_enabled():
        return
    _get_model(
        model or config.WHISPER_MODEL,
        device or config.WHISPER_DEVICE,
        compute_type or config.WHISPER_COMPUTE_TYPE,
    )


def transcribe_audio(
    path: str,
    *,
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> str:
    """Transcribe an audio file to text.

    Accepts WAV (as produced by the recorder) plus any format ffmpeg can
    decode (mp3, m4a, webm, ogg, ...). Returns the joined transcript with
    whitespace normalised.

    Decoding is tuned for speech transcripts:
      - ``vad_filter`` runs Silero VAD to drop silence, which suppresses the
        repeated/hallucinated phrases Whisper emits over quiet stretches and
        sharpens segment boundaries.
      - ``initial_prompt`` primes the decoder with well-punctuated text so the
        output mirrors that style (better sentence boundaries downstream).
      - ``condition_on_previous_text`` lets each chunk be decoded conditioned
        on the prior transcript — Whisper's built-in cross-segment context.

    ``model``/``device``/``compute_type`` override the config defaults (the UI
    exposes the first and last so users can trade speed for accuracy).

    With ``GROQ_API_KEY`` set, transcription goes to Groq's hosted Whisper
    instead (no local model, fast, ~zero RAM) — same idiom as the NVIDIA →
    edge-tts pair in ``tts.py``.
    """
    if groq_is_enabled():
        return _transcribe_groq(path)
    whisper = _get_model(
        model or config.WHISPER_MODEL,
        device or config.WHISPER_DEVICE,
        compute_type or config.WHISPER_COMPUTE_TYPE,
    )
    segments, _info = whisper.transcribe(
        path,
        language="en",
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=True,
        initial_prompt=config.WHISPER_INITIAL_PROMPT,
        condition_on_previous_text=True,
    )
    text = " ".join(segment.text.strip() for segment in segments)
    return " ".join(text.split())
