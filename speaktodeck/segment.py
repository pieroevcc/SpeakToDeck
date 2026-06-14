"""Split a transcript into clean individual sentences.

Two free, local backends:
  - "sat"   : wtpsplit SaT model. Neural and punctuation-agnostic, so it copes
              with the weak/missing punctuation typical of speech transcripts.
  - "pysbd" : rule-based, zero model download, instant. Great when the text is
              already well punctuated.
"""

from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache

from . import config

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_sat():
    from wtpsplit import SaT

    return SaT(config.SAT_MODEL)


def _sanitize(text: str) -> str:
    """Drop invisible control/format characters and normalize whitespace.

    Whisper transcripts occasionally carry zero-width spaces, BOMs, joiners,
    or stray control bytes. These are invisible but destabilize the SaT neural
    tokenizer's chunking math (it can raise deep inside wtpsplit, e.g. an
    ``int()``/``NoneType`` error while assembling token chunks). Stripping the
    Unicode "C" category (control/format/surrogate/private-use/unassigned) and
    collapsing whitespace gives the splitter clean input to work with.
    """
    text = unicodedata.normalize("NFC", text)
    cleaned = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    return " ".join(cleaned.split())


@lru_cache(maxsize=1)
def _get_pysbd():
    import pysbd

    return pysbd.Segmenter(language="en", clean=True)


def _clean(sentences: list[str]) -> list[str]:
    """Trim, drop empties, and de-duplicate while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in sentences:
        s = " ".join(raw.split())
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def preload(backend: str | None = None) -> None:
    """Load the segmentation backend model ahead of time.

    Called at app startup (in a background thread) so the first split isn't
    paying the model-load cost. The underlying getters are ``lru_cache``-d, so
    repeated calls are no-ops.
    """
    backend = backend or config.SENTENCE_BACKEND
    if backend == "sat":
        _get_sat()
    elif backend == "pysbd":
        _get_pysbd()


def split_sentences(text: str, backend: str | None = None) -> list[str]:
    """Return a list of clean sentences from ``text``."""
    text = (text or "").strip()
    if not text:
        return []

    backend = backend or config.SENTENCE_BACKEND
    if backend == "sat":
        text = _sanitize(text)
        if not text:
            return []
        try:
            sentences = list(_get_sat().split(text))
        except Exception:
            # wtpsplit can fail on degenerate input or stack incompatibilities.
            # Never let segmentation crash the pipeline — fall back to pysbd.
            log.warning("SaT backend failed; falling back to pysbd", exc_info=True)
            sentences = _get_pysbd().segment(text)
    elif backend == "pysbd":
        sentences = _get_pysbd().segment(text)
    else:
        raise ValueError(f"Unknown sentence backend: {backend!r}")

    return _clean(sentences)
