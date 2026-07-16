"""Central configuration for SpeakToDeck.

Tweak model sizes, the sentence-splitting backend, language list, and service
endpoints here. Nothing in this module performs I/O at import time.
"""

from __future__ import annotations

import os

# --- Transcription (faster-whisper) ------------------------------------------
# "small.en" is an English-only model: more accurate than the multilingual
# "base" at a similar size, with cleaner punctuation (which drives sentence
# quality downstream). Bump to "medium.en" for more accuracy at a CPU-speed
# cost, or "base"/"small" for multilingual. Downloaded once, then cached.
WHISPER_MODEL = "large-v3-turbo"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8" 
WHISPER_BEAM_SIZE = 1  # beam search width — 1 (greedy) is ~1.5-2x faster than 5
# with negligible quality loss on clean English speech; raise toward 5 if you
# transcribe noisy audio and want the decoder to weigh more alternatives.
# Priming text fed to Whisper's decoder. Whisper mirrors the style of this
# prompt, so well-punctuated, capitalized full sentences nudge it to emit the
# same — which gives the sentence splitter clean boundaries to work with.
WHISPER_INITIAL_PROMPT = (
    "Hello. Here is a clear transcription with proper capitalization, "
    "punctuation, and complete sentences."
)

# Choices surfaced in the sidebar (the UI overrides the defaults above per
# session). ``.en`` models are English-only and more accurate for English; the
# rest are multilingual. Larger = more accurate, slower, bigger one-time download.
WHISPER_MODEL_CHOICES = [
    "tiny.en", "base.en", "small.en", "medium.en", "large-v3", "large-v3-turbo",
]
# Compute types valid on CPU (where faster-whisper runs on Apple Silicon — it
# has no Metal/GPU backend, so Apple users stay on CPU and tune this instead).
# int8 = fastest/smallest, float32 = most accurate/slowest, int8_float32 between.
WHISPER_COMPUTE_TYPE_CHOICES = ["int8", "int8_float32", "float32"]

# --- Transcription: Groq API (optional, replaces local Whisper) ---------------
# When GROQ_API_KEY is set, transcription goes to Groq's hosted Whisper instead
# of loading a local model — required on small hosts (Streamlit Cloud ~1 GB RAM)
# and much faster everywhere. No key -> local faster-whisper, unchanged.
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# --- Sentence segmentation ----------------------------------------------------
# "sat"  -> wtpsplit SaT model (neural, punctuation-agnostic, best for speech)
# "pysbd"-> rule-based, zero download, fast (best when text is well punctuated)
# Env-overridable so small hosts can force pysbd (SaT pulls in torch, ~1 GB RAM).
SENTENCE_BACKEND = os.environ.get("SENTENCE_BACKEND", "sat")
SAT_MODEL = "sat-3l-sm"

# --- Translation target languages (label -> ISO code used by deep-translator) -
# Codes also drive edge-tts voice selection (see EDGE_TTS_VOICES below).
LANGUAGES: dict[str, str] = {
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Korean": "ko",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Dutch": "nl",
}
DEFAULT_LANGUAGE = "Spanish"

# --- Anki delivery ------------------------------------------------------------
ANKI_CONNECT_URL = "http://localhost:8765"
ANKI_CONNECT_VERSION = 6
DEFAULT_DECK_NAME = "SpeakToDeck"
ANKI_NOTE_MODEL = "Basic"  # ships with every Anki install; fields Front/Back

# --- Pronunciation audio: NVIDIA Magpie TTS (primary engine) ------------------
# NVIDIA's hosted multilingual neural TTS, reached over Riva gRPC and
# authenticated with NVIDIA_API_KEY. Only the languages Magpie supports are
# mapped below; every other target language falls back to edge-tts (and so does
# any NVIDIA failure). See speaktodeck/tts.py.
NVIDIA_TTS_SERVER = "grpc.nvcf.nvidia.com:443"
NVIDIA_TTS_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"
NVIDIA_TTS_SAMPLE_RATE = 44100  # Hz; Magpie returns 16-bit mono PCM
# Target ISO code -> (Magpie voice_name, Riva language_code).
NVIDIA_TTS_VOICES: dict[str, tuple[str, str]] = {
    "es": ("Magpie-Multilingual.ES-US.Isabela", "es-US"),
    "fr": ("Magpie-Multilingual.FR-FR.Louise", "fr-FR"),
    "de": ("Magpie-Multilingual.DE-DE.Mia", "de-DE"),
    "it": ("Magpie-Multilingual.IT-IT.Isabela", "it-IT"),
    "zh-CN": ("Magpie-Multilingual.ZH-CN.Mia", "zh-CN"),
    "hi": ("Magpie-Multilingual.HI-IN.Aria", "hi-IN"),
    "ja": ("Magpie-Multilingual.JA-JP.Mia", "ja-JP"),
}

# --- Pronunciation audio: edge-tts (fallback engine) --------------------------
# Microsoft Edge neural voices: free, no API key, and covers every target
# language. One natural voice per language (keyed by the ISO code from LANGUAGES
# above). Used when NVIDIA can't voice a language or its call fails.
EDGE_TTS_VOICES: dict[str, str] = {
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-PT-RaquelNeural",
    "ko": "ko-KR-SunHiNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh-CN": "zh-CN-XiaoxiaoNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "hi": "hi-IN-SwaraNeural",
    "nl": "nl-NL-ColetteNeural",
}
