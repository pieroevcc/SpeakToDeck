"""Translate English sentences into a target language (free, no API key).

Uses deep-translator's GoogleTranslator (Google's free web endpoint). This
module is pure with respect to ``target_lang`` — switching the language in the
UI simply re-runs ``translate_sentences`` with a new code.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Cap on concurrent translation requests. The work is network-bound (Google's
# web endpoint), so threads overlap latency cheaply; keep it modest to stay
# polite to the free endpoint and avoid tripping its rate limiter.
_MAX_WORKERS = 8


@dataclass
class TranslationResult:
    """One translated sentence. ``ok=False`` means the original English was
    kept because translation failed (e.g. rate limit / network)."""

    source: str
    text: str
    ok: bool = True


def translate_sentences(
    sentences: list[str], target_lang: str, source_lang: str = "en"
) -> list[TranslationResult]:
    """Translate each sentence into ``target_lang``.

    Failures degrade gracefully per-sentence: the English text is kept and the
    result is flagged ``ok=False`` rather than aborting the whole batch.

    Sentences are translated concurrently (the work is network-bound) but the
    returned list preserves input order.
    """
    from deep_translator import GoogleTranslator

    if not sentences:
        return []

    def _one(sentence: str) -> TranslationResult:
        # A fresh translator per call keeps the worker threads independent (the
        # GoogleTranslator object isn't designed to be shared across threads).
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(sentence) or sentence
            return TranslationResult(source=sentence, text=translated, ok=True)
        except Exception:
            return TranslationResult(source=sentence, text=sentence, ok=False)

    workers = min(_MAX_WORKERS, len(sentences))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # executor.map preserves input order while overlapping the requests.
        return list(pool.map(_one, sentences))
