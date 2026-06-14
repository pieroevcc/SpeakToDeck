"""Flashcard candidate model and builder.

One candidate per sentence: English on the front, the translation on the back,
with optional pronunciation audio (bytes) for the translation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["pending", "accepted", "discarded"]


@dataclass
class Candidate:
    english: str
    translation: str
    audio: bytes | None = None
    audio_ext: str = "mp3"  # container of ``audio`` bytes: "mp3" (edge) or "wav" (NVIDIA)
    translation_ok: bool = True
    status: Status = "pending"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


def build_candidates(
    english: list[str],
    translations: list[str],
    audio: list[bytes | None] | None = None,
    audio_ext: list[str] | None = None,
    translation_ok: list[bool] | None = None,
) -> list[Candidate]:
    """Zip parallel lists into a list of pending ``Candidate`` objects.

    ``audio``, ``audio_ext`` and ``translation_ok`` are optional and default to
    None/"mp3"/True. Raises ValueError if ``english`` and ``translations`` differ
    in length.
    """
    if len(english) != len(translations):
        raise ValueError(
            f"english ({len(english)}) and translations ({len(translations)}) "
            "must be the same length"
        )

    audio = audio or [None] * len(english)
    audio_ext = audio_ext or ["mp3"] * len(english)
    translation_ok = translation_ok or [True] * len(english)

    return [
        Candidate(
            english=en,
            translation=tr,
            audio=au,
            audio_ext=ext,
            translation_ok=ok,
        )
        for en, tr, au, ext, ok in zip(
            english, translations, audio, audio_ext, translation_ok
        )
    ]


def accepted(candidates: list[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.status == "accepted"]
