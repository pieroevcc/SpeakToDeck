"""SpeakToDeck — Streamlit dashboard.

Pipeline: record/import audio -> transcribe (Whisper) -> split into sentences
-> translate -> optional TTS -> curate (accept/discard) -> send to Anki desktop
(AnkiConnect) with a .apkg download fallback.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import os

# Quiet third-party noise before the lazy model loads fire. Whisper/wtpsplit
# pull weights from the HF Hub anonymously (no token needed) and wtpsplit calls
# a deprecated transformers arg internally — none of which the user can act on.
# NOTE: the "unauthenticated requests to the HF Hub" line is written by native
# code straight to fd 2 (no Python hook or env flag silences it); it only shows
# while models are downloading, so it disappears once the cache is warm.
# Disable the Xet download backend (hf-xet). Some HF repos (e.g. the large-v3
# / large-v3-turbo CT2 weights) are Xet-backed, and the Xet client throws
# "[WinError 10038] ... not a socket" mid-download on Windows. Forcing the
# classic HTTPS path makes those downloads reliable; cached models are unaffected.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_UPDATE_CHECK", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import tempfile

import streamlit as st
from dotenv import load_dotenv

from speaktodeck import (
    anki_connect,
    anki_export,
    config,
    flashcards,
    segment,
    transcribe,
    translate,
    tts,
)

load_dotenv()  # pick up NVIDIA_API_KEY from .env if present

st.set_page_config(page_title="SpeakToDeck", page_icon="🎙️", layout="centered")


# --- session state defaults ---------------------------------------------------
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("candidates", [])          # list[flashcards.Candidate]
    ss.setdefault("sentences", [])           # cached English sentences
    ss.setdefault("target_lang_label", config.DEFAULT_LANGUAGE)
    ss.setdefault("backend", config.SENTENCE_BACKEND)
    ss.setdefault("whisper_model", config.WHISPER_MODEL)
    ss.setdefault("whisper_compute_type", config.WHISPER_COMPUTE_TYPE)
    ss.setdefault("use_tts", True)  # edge-tts needs no key, so audio is always available
    ss.setdefault("deck_name", config.DEFAULT_DECK_NAME)
    ss.setdefault("apkg_bytes", None)
    ss.setdefault("recorder_key", 0)         # bump to reset the audio_input widget


_init_state()


@st.cache_resource(show_spinner=False)
def _warm_models() -> bool:
    """Preload Whisper + the sentence splitter in a background thread so the
    first 'Generate flashcards' isn't a cold model load. ``st.cache_resource``
    makes this run exactly once per process; the thread is a daemon so it never
    holds up shutdown, and failures are swallowed (the on-demand load in the
    pipeline will surface any real error)."""
    import threading

    def _load() -> None:
        try:
            transcribe.preload(
                model=config.WHISPER_MODEL,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
        except Exception:  # noqa: BLE001 - best-effort warm-up
            pass
        try:
            segment.preload(config.SENTENCE_BACKEND)
        except Exception:  # noqa: BLE001 - best-effort warm-up
            pass

    threading.Thread(target=_load, name="speaktodeck-warmup", daemon=True).start()
    return True


_warm_models()


def _target_code() -> str:
    return config.LANGUAGES[st.session_state.target_lang_label]


def _clear_recording() -> None:
    """Drop the recorded audio and everything derived from it. Bumping the
    recorder key forces a fresh st.audio_input widget (its buffer can't be
    cleared in place), and we wipe the cached sentences/candidates/deck."""
    st.session_state.recorder_key += 1
    st.session_state.sentences = []
    st.session_state.candidates = []
    st.session_state.apkg_bytes = None


def _retranslate() -> None:
    """Re-run translation (and TTS) for the cached sentences after a settings
    change, preserving accept/discard decisions where sentences are unchanged."""
    sentences = st.session_state.sentences
    if not sentences:
        return
    code = _target_code()
    results = translate.translate_sentences(sentences, code)

    prev_status = {c.english: c.status for c in st.session_state.candidates}
    if st.session_state.use_tts:
        tts_results = tts.synthesize_many([r.text for r in results], code)
    else:
        tts_results = [None] * len(results)
    audio = [res[0] if res else None for res in tts_results]
    audio_ext = [res[1] if res else "mp3" for res in tts_results]

    cards = flashcards.build_candidates(
        english=[r.source for r in results],
        translations=[r.text for r in results],
        audio=audio,
        audio_ext=audio_ext,
        translation_ok=[r.ok for r in results],
    )
    for c in cards:
        if c.english in prev_status:
            c.status = prev_status[c.english]
    st.session_state.candidates = cards
    st.session_state.apkg_bytes = None


# --- sidebar / settings (Request C) ------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    new_label = st.selectbox(
        "Translate into",
        options=list(config.LANGUAGES.keys()),
        index=list(config.LANGUAGES.keys()).index(st.session_state.target_lang_label),
        help="Switch the target language anytime. Existing cards re-translate in place.",
    )
    if new_label != st.session_state.target_lang_label:
        st.session_state.target_lang_label = new_label
        _retranslate()

    st.session_state.backend = st.radio(
        "Sentence splitter",
        options=["sat", "pysbd"],
        index=["sat", "pysbd"].index(st.session_state.backend),
        format_func=lambda b: "wtpsplit (best for speech)" if b == "sat" else "pysbd (fast)",
    )

    st.session_state.use_tts = st.checkbox(
        "Add pronunciation audio",
        value=st.session_state.use_tts,
        help="Synthesizes the translated sentence so cards carry native "
        "pronunciation audio — NVIDIA Magpie neural TTS when NVIDIA_API_KEY is "
        "set (free Microsoft Edge voices otherwise / as fallback).",
    )

    if tts.nvidia_is_enabled():
        st.caption("🔊 Pronunciation: NVIDIA Magpie TTS (edge-tts fallback).")
    else:
        st.caption("💡 Set NVIDIA_API_KEY to use NVIDIA Magpie TTS (edge-tts is used otherwise).")

    with st.expander("🧠 Transcription model"):
        st.session_state.whisper_model = st.selectbox(
            "Whisper model",
            options=config.WHISPER_MODEL_CHOICES,
            index=config.WHISPER_MODEL_CHOICES.index(st.session_state.whisper_model)
            if st.session_state.whisper_model in config.WHISPER_MODEL_CHOICES
            else config.WHISPER_MODEL_CHOICES.index("small.en"),
            help="Larger = more accurate, slower, bigger one-time download. "
            "`.en` models are English-only and best for English audio.",
        )
        st.session_state.whisper_compute_type = st.selectbox(
            "Compute type",
            options=config.WHISPER_COMPUTE_TYPE_CHOICES,
            index=config.WHISPER_COMPUTE_TYPE_CHOICES.index(
                st.session_state.whisper_compute_type
            )
            if st.session_state.whisper_compute_type in config.WHISPER_COMPUTE_TYPE_CHOICES
            else 0,
            help="Speed/accuracy trade-off on CPU: `int8` is fastest, `float32` "
            "most accurate. Apple Silicon runs on CPU (no GPU backend), so this "
            "is the main accuracy dial for Mac users.",
        )
        st.caption("Switching reloads the model on the next run (first use downloads it).")

    st.divider()
    connected, anki_message = anki_connect.status()
    if connected:
        st.success(anki_message)
    else:
        st.warning(anki_message)


# --- main ---------------------------------------------------------------------
st.title("🎙️ SpeakToDeck")
st.caption("Speak or import English audio → translated Anki flashcards.")

def _text_pipeline(text: str) -> None:
    """Everything after transcription: split → translate → TTS → candidates.
    Shared by the audio path and the paste-text path."""
    code = _target_code()
    with st.status("Building flashcards…", expanded=True) as status:
        st.write("Splitting into sentences…")
        sentences = segment.split_sentences(text, backend=st.session_state.backend)
        st.session_state.sentences = sentences

        st.write(f"Translating {len(sentences)} sentence(s) → {st.session_state.target_lang_label}…")
        results = translate.translate_sentences(sentences, code)

        audio_list: list[bytes | None] = [None] * len(results)
        audio_ext_list: list[str] = ["mp3"] * len(results)
        if st.session_state.use_tts:
            st.write("Synthesizing pronunciation audio…")
            for i, result in enumerate(tts.synthesize_many([r.text for r in results], code)):
                if result:
                    audio_list[i], audio_ext_list[i] = result

        st.session_state.candidates = flashcards.build_candidates(
            english=[r.source for r in results],
            translations=[r.text for r in results],
            audio=audio_list,
            audio_ext=audio_ext_list,
            translation_ok=[r.ok for r in results],
        )
        st.session_state.apkg_bytes = None
        status.update(label=f"Created {len(results)} candidate card(s)", state="complete")

    # Rerun so the freshly-built candidates render this pass (the tab bodies and
    # card counts were already computed from session state earlier in the script).
    st.rerun()


def _run_pipeline(audio) -> None:
    suffix = "." + (getattr(audio, "name", "audio.wav").rsplit(".", 1)[-1])
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio.getvalue())
        tmp_path = tmp.name

    with st.spinner(f"Transcribing audio (Whisper · {st.session_state.whisper_model})…"):
        text = transcribe.transcribe_audio(
            tmp_path,
            model=st.session_state.whisper_model,
            compute_type=st.session_state.whisper_compute_type,
        )
    if not text.strip():
        st.error("No speech detected")
        return
    _text_pipeline(text)


def _render_card_body(card) -> None:
    """English + translation + audio player — shared by both card lists."""
    st.markdown(f"**{card.english}**")
    if card.translation_ok:
        st.markdown(f":blue[{card.translation}]")
    else:
        st.markdown(
            f":orange[{card.translation}] _(translation failed — kept English)_"
        )
    if card.audio:
        st.audio(
            card.audio,
            format="audio/wav" if card.audio_ext == "wav" else "audio/mpeg",
        )


candidates = st.session_state.candidates
accepted_cards = flashcards.accepted(candidates)

tab_create, tab_cards = st.tabs(
    ["🎙️ Create", f"🗂️ My cards ({len(accepted_cards)})"]
)

# --- Create tab: add audio + review candidates -------------------------------
with tab_create:
    st.subheader("1. Add audio")
    col_rec, col_imp = st.columns(2)
    with col_rec:
        recorded = st.audio_input("Record", key=f"recorder_{st.session_state.recorder_key}")
    with col_imp:
        uploaded = st.file_uploader(
            "Import audio", type=["wav", "mp3", "m4a", "webm", "ogg", "mpeg"]
        )
    audio_file = recorded or uploaded

    b_gen, b_clear = st.columns([3, 1])
    if b_gen.button(
        "Generate flashcards", type="primary", disabled=audio_file is None, use_container_width=True
    ):
        _run_pipeline(audio_file)
    if b_clear.button(
        "🗑️ Clear", disabled=recorded is None, use_container_width=True,
        help="Discard the recorded audio and any cards generated from it.",
    ):
        _clear_recording()
        st.rerun()

    with st.expander("📋 Or paste text instead"):
        pasted = st.text_area(
            "Text to turn into cards",
            placeholder="Paste English text — it skips transcription and goes "
            "straight to sentence splitting.",
            label_visibility="collapsed",
        )
        if st.button(
            "Generate from text", type="primary", disabled=not pasted.strip(),
            use_container_width=True,
        ):
            _text_pipeline(pasted)

    if candidates:
        pending = [c for c in candidates if c.status == "pending"]
        st.subheader("2. Review candidates")
        st.caption(
            f"✅ {len(accepted_cards)} accepted · ⏳ {len(pending)} remaining"
        )
        if not pending:
            st.success("All candidates reviewed — see the **My cards** tab.")
        for card in pending:
            with st.container(border=True):
                _render_card_body(card)
                b_accept, b_discard = st.columns(2)
                if b_accept.button("✅ Accept", key=f"a_{card.id}", use_container_width=True):
                    card.status = "accepted"
                    st.rerun()
                if b_discard.button("❌ Discard", key=f"d_{card.id}", use_container_width=True):
                    card.status = "discarded"
                    st.rerun()

# --- My cards tab: selected cards + deck delivery ----------------------------
with tab_cards:
    st.subheader("Selected cards")
    if not accepted_cards:
        st.info("No cards selected yet. Accept candidates in the **Create** tab.")
    else:
        st.caption(f"{len(accepted_cards)} card(s) ready.")
        for card in accepted_cards:
            with st.container(border=True):
                _render_card_body(card)
                if st.button("↩️ Remove", key=f"rm_{card.id}"):
                    card.status = "pending"
                    st.session_state.apkg_bytes = None
                    st.rerun()

        st.divider()
        st.subheader("Build your deck")
        st.session_state.deck_name = st.text_input(
            "Deck name", value=st.session_state.deck_name
        )

        if st.button("📤 Send to Anki", type="primary"):
            try:
                added = anki_connect.push_cards(accepted_cards, st.session_state.deck_name)
                st.toast(f"📤 Deck sent to Anki — '{st.session_state.deck_name}'!", icon="✅")
                st.success(
                    f"Your deck is on its way! Added {added} card(s) to "
                    f"'{st.session_state.deck_name}' — open Anki to start studying."
                )
                st.balloons()
            except anki_connect.AnkiConnectError as exc:
                st.error(f"{exc}")
                st.info("Use the .apkg download below instead.")
                st.session_state.apkg_bytes = anki_export.export_deck(
                    accepted_cards, st.session_state.deck_name
                )

        if st.session_state.apkg_bytes is None:
            if st.button("Build .apkg download"):
                st.session_state.apkg_bytes = anki_export.export_deck(
                    accepted_cards, st.session_state.deck_name
                )

        if st.session_state.apkg_bytes:
            st.download_button(
                "⬇️ Download .apkg",
                data=st.session_state.apkg_bytes,
                file_name=f"{st.session_state.deck_name}.apkg",
                mime="application/octet-stream",
            )
