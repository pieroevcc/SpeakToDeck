# 🎙️ SpeakToDeck

Turn spoken English audio into translated **Anki flashcards** — mostly on free
and local tooling. Record or import audio, and SpeakToDeck transcribes it, splits
it into clean sentences, translates each one into a language you choose, lets you
accept/discard candidates, then pushes the deck straight into your open Anki app.

## Pipeline

```
record / import audio
  → transcribe        (faster-whisper, local, offline)
  → split sentences   (wtpsplit SaT, or pysbd)
  → translate         (deep-translator / Google free endpoint)
  → pronunciation TTS  (NVIDIA Magpie neural TTS → edge-tts fallback)
  → accept / discard   (curation UI)
  → send to Anki       (AnkiConnect) — or download .apkg
```

## Setup

### 1. Python deps
```powershell
pip install -r requirements.txt
```

### 2. ffmpeg (required by Whisper to decode non-WAV uploads)
```powershell
winget install Gyan.FFmpeg
```
Confirm with `ffmpeg -version`.

### 3. Anki + AnkiConnect (for "Send to Anki")
1. Install [Anki desktop](https://apps.ankiweb.net/).
2. In Anki: **Tools → Add-ons → Get Add-ons…** and enter code **`2055492159`**.
3. Restart Anki. Visit <http://localhost:8765> in a browser — it should say
   `AnkiConnect`. Keep Anki open while using SpeakToDeck.

> No Anki running? SpeakToDeck falls back to a downloadable `.apkg` file you can
> import manually (**File → Import**).

### 4. Pronunciation audio (NVIDIA primary, edge-tts fallback)

Each accepted card carries native pronunciation audio of the translated
sentence, synthesized by one of two engines (toggle in the sidebar):

- **Primary — NVIDIA Magpie multilingual TTS.** High-quality hosted neural
  voices, reached over Riva gRPC (the `nvidia-riva-client` dependency) and
  authenticated with your `NVIDIA_API_KEY`. It covers Spanish, French, German,
  Italian, Mandarin, Hindi, and Japanese; voices live in
  [`speaktodeck/config.py`](speaktodeck/config.py) (`NVIDIA_TTS_VOICES`).
- **Fallback — [`edge-tts`](https://github.com/rany2/edge-tts).** Free, key-less
  Microsoft Edge neural voices that cover **every** target language
  (`EDGE_TTS_VOICES`). Used automatically when no NVIDIA key is set, the language
  isn't in Magpie's set, or an NVIDIA call fails.

Set your key in `.env` (copy from `.env.example`) to enable the NVIDIA engine:
```
NVIDIA_API_KEY=nvapi-...
```

> TTS is **best-effort**: both engines stream over the network, so it needs
> internet. Any per-sentence failure degrades gracefully — NVIDIA failures fall
> back to edge-tts, and an edge-tts failure simply creates that card without
> audio rather than breaking the run.

## Run

```powershell
streamlit run app.py
```
Opens at <http://localhost:8501>.

## Usage

1. **⚙️ Settings** (sidebar): pick the target language (switchable anytime — cards
   re-translate in place), choose the sentence splitter, and toggle pronunciation
   audio (NVIDIA Magpie TTS with an `NVIDIA_API_KEY` set, edge-tts otherwise).
   Under **🧠 Transcription model** you can switch the Whisper model and compute
   type per session (see below).
2. **🎙️ Create** tab → **Add audio**: click **Record** and speak, or **Import** a file.
3. **Generate flashcards**: runs the pipeline and lists candidates.
4. **Review**: **✅ Accept** / **❌ Discard** each candidate (accepted ones move to
   the **My cards** tab).
5. **🗂️ My cards** tab: review everything you selected (with audio), **↩️ Remove**
   any you change your mind on, name the deck, then **📤 Send to Anki** (or download
   the `.apkg`).

## Configuration

Everything tunable lives in [`speaktodeck/config.py`](speaktodeck/config.py): Whisper
model size, beam size and decoder priming prompt, sentence backend, the language
list, AnkiConnect URL, the NVIDIA Magpie TTS voices/endpoint, and the
per-language edge-tts fallback voices.

Transcription is tuned for speech in [`speaktodeck/transcribe.py`](speaktodeck/transcribe.py):
voice-activity filtering (drops silence to suppress hallucinated repeats), a
well-punctuated `initial_prompt` (nudges Whisper toward clean sentences), and
cross-segment context conditioning.

### Whisper model & compute type (sidebar)

The **🧠 Transcription model** expander lets you change two settings live, per
session, without editing config:

- **Whisper model** — `tiny.en` → `large-v3`. Larger is more accurate but slower
  and a bigger one-time download; `.en` models are English-only and best for
  English audio. Switching reloads the model on the next run (a few stay cached
  warm). Defaults and the choice list live in `config.py`
  (`WHISPER_MODEL`, `WHISPER_MODEL_CHOICES`).
- **Compute type** — `int8` (fastest) → `float32` (most accurate), with
  `int8_float32` in between.

> **Apple Silicon / Mac users:** faster-whisper runs on **CPU** on macOS (no
> Metal/GPU backend), so the compute type is your main accuracy dial — try
> `float32` if you want better transcripts and can spare the speed. The compute
> options are restricted to CPU-valid types on purpose (GPU-only types like
> `float16` would error on a Mac).

## Tests

```powershell
pytest
```
Offline tests run by default. Tests needing network/model downloads are marked
`network` and skipped (`pytest -m network` to include them).

## Tech stack

| Stage | Tooling |
| --- | --- |
| UI | [Streamlit](https://streamlit.io/) |
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper, local) + [ffmpeg](https://ffmpeg.org/) for decoding |
| Sentence splitting | [wtpsplit](https://github.com/segment-any-text/wtpsplit) (SaT) / [pysbd](https://github.com/nipunsadvilkar/pySBD) |
| Translation | [deep-translator](https://github.com/nidhaloff/deep-translator) (Google free endpoint) |
| Pronunciation TTS | [NVIDIA Magpie multilingual](https://build.nvidia.com/nvidia/magpie-tts-multilingual) (Riva gRPC, `nvidia-riva-client`) → [edge-tts](https://github.com/rany2/edge-tts) fallback |
| Flashcards / Anki | [AnkiConnect](https://foosoft.net/projects/anki-connect/) (live) · [genanki](https://github.com/kerrickstaley/genanki) (`.apkg` export) |
| Config / secrets | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| Tests | [pytest](https://docs.pytest.org/) |

## Notes

- First run downloads the Whisper and wtpsplit models once, then caches them.
  (You may see a one-time "unauthenticated requests to the HF Hub" line during
  that download — it's harmless and stops once the models are cached.)
- `deep-translator` uses Google's unofficial free endpoint (rate-limited). If it
  breaks, `argostranslate` is a fully-offline alternative.
- The default Whisper model (`WHISPER_MODEL` in `config.py`) trades accuracy for
  CPU speed: larger models are more accurate but slower per clip and a bigger
  one-time download — drop to a smaller one if you need speed over accuracy.
- NVIDIA Magpie TTS covers only 7 of the target languages (Spanish, French,
  German, Italian, Mandarin, Hindi, Japanese); the rest (Portuguese, Korean,
  Russian, Arabic, Dutch) always use the edge-tts fallback. No key set → edge-tts
  for everything.
