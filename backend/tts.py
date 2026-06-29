"""nChat TTS — local Kokoro wrapper.

Self-hosted text-to-speech for nChat. Caches one ``KPipeline`` per language code
(loading is expensive — hold it resident, the same instinct as Ollama's
``keep_alive``), dispatches blocking synthesis to a threadpool from the caller,
and returns WAV bytes.

The real value-add over a raw Kokoro call is :func:`markdown_to_speech`. Chat
output is Markdown, and a TTS engine read raw will narrate backticks and spell
out code blocks letter by letter. The cleaner strips to prose and, by default,
*announces* code ("Code block, twelve lines.") rather than reading it.

Design contract (from the nChat voice-loop design):

* **Model cached once, blocking inference dispatched to a thread, text in /
  bytes out.** Identical shape to ``stt.py``.
* **Lazy, guarded engine import.** Importing this module never requires Kokoro
  or torch. If the runtime/weights are absent, :meth:`KokoroTTS.synthesize`
  raises :class:`TTSUnavailable` and the caller degrades. Delete this file and
  its routes and nChat is exactly what it is today.
* **``device="auto"``.** On an Apple-Silicon Mac with no CUDA this resolves to
  CPU, which is what we want — Kokoro at 82M runs faster than real time on
  CPU there, leaving the Metal cores to Ollama.

The :func:`markdown_to_speech` cleaner and the voice catalog are stdlib-only and
fully testable without the engine present (Phase 0 gate). Run ``python -m
backend.tts`` for the isolation harness.
"""
from __future__ import annotations

import io
import re
import wave
import struct
import threading

# Kokoro's native sample rate. Fixed by the model; do not change.
SAMPLE_RATE = 24000


class TTSUnavailable(RuntimeError):
    """Raised when the Kokoro runtime or its weights are not present.

    The endpoint layer catches this and returns a clear 5xx; the rest of nChat
    keeps working. This is the 'degrade, never fail' contract for the TTS link.
    """


# --- Voice catalog -----------------------------------------------------------
#
# Kokoro voice ids encode language in their prefix: ``a`` = American English,
# ``b`` = British English. The first letter after that is gender (f/m). We keep
# a small curated catalog rather than the full set so the dropdown stays usable;
# extend freely — the engine accepts any valid Kokoro voice id.

VOICES: list[dict] = [
    {"id": "af_heart",    "label": "Heart (US, female)",      "lang": "a"},
    {"id": "af_bella",    "label": "Bella (US, female)",      "lang": "a"},
    {"id": "af_sarah",    "label": "Sarah (US, female)",      "lang": "a"},
    {"id": "af_nicole",   "label": "Nicole (US, female)",     "lang": "a"},
    {"id": "am_adam",     "label": "Adam (US, male)",         "lang": "a"},
    {"id": "am_michael",  "label": "Michael (US, male)",      "lang": "a"},
    {"id": "am_fenrir",   "label": "Fenrir (US, male)",       "lang": "a"},
    {"id": "bf_emma",     "label": "Emma (UK, female)",       "lang": "b"},
    {"id": "bf_isabella", "label": "Isabella (UK, female)",   "lang": "b"},
    {"id": "bm_george",   "label": "George (UK, male)",       "lang": "b"},
    {"id": "bm_lewis",    "label": "Lewis (UK, male)",        "lang": "b"},
]

_VOICE_IDS = {v["id"] for v in VOICES}
DEFAULT_VOICE = "af_heart"


def list_voices() -> list[dict]:
    """Return the voice catalog for the frontend dropdown. Stdlib-only."""
    return list(VOICES)


def lang_for_voice(voice: str) -> str:
    """Derive Kokoro's lang_code from a voice id's prefix.

    Falls back to American English ('a') for unknown ids rather than failing —
    a wrong-but-working voice beats a hard error mid-read.
    """
    return voice[0] if voice and voice[0] in ("a", "b") else "a"


# --- The cleaner (the value-add) ---------------------------------------------

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_LIST_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_ORDERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)(.+?)\1", re.DOTALL)
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def _announce_code(lang: str, body: str) -> str:
    """Turn a fenced code block into a spoken announcement instead of reading it.

    A TTS engine reading a code block aloud is unlistenable. We say what it is —
    language and length — and move on. This is the default; ``read_code=True``
    keeps the raw code in the speech text instead.
    """
    lang = (lang or "").strip()
    n_lines = len([ln for ln in body.splitlines() if ln.strip()])
    plural = "line" if n_lines == 1 else "lines"
    if lang:
        return f" (Code block in {lang}, {n_lines} {plural}.) "
    return f" (Code block, {n_lines} {plural}.) "


def markdown_to_speech(md: str, read_code: bool = False) -> str:
    """Convert chat Markdown into a clean, speakable string.

    Strips fenced/inline code (announcing fenced blocks unless ``read_code``),
    images, link targets (keeping link text), headers, emphasis markers, list
    bullets, blockquote markers, horizontal rules, and tables. Collapses
    whitespace. Returns ``""`` for input that has nothing speakable (e.g. a
    code-only answer with ``read_code=False``) — the caller decides what to do
    with empty (we synthesize a short "nothing to read").

    Pure stdlib, no engine required — this is the part the Phase 0 fixture gate
    proves.
    """
    if not md:
        return ""

    text = md

    # Fenced code blocks first (before inline), so their backticks don't leak.
    if read_code:
        text = _FENCE_RE.sub(lambda m: f"\n{m.group(2)}\n", text)
    else:
        text = _FENCE_RE.sub(lambda m: _announce_code(m.group(1), m.group(2)), text)

    # Any unterminated/odd fence leftovers: drop stray triple-backticks.
    text = text.replace("```", " ")

    # Images -> alt text (or nothing); links -> their visible text.
    text = _IMAGE_RE.sub(lambda m: m.group(1) or "", text)
    text = _LINK_RE.sub(lambda m: m.group(1), text)

    # Inline code -> its contents, read as words (short, usually a name/flag).
    text = _INLINE_CODE_RE.sub(lambda m: m.group(1), text)

    # Tables: announce rather than read pipes and dashes.
    if _TABLE_ROW_RE.search(text):
        # Drop separator rows (|---|---|) entirely, flatten data rows to commas.
        def _table_line(m: re.Match) -> str:
            cells = [c.strip() for c in m.group(0).strip().strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                return ""  # separator row
            return ", ".join(c for c in cells if c)
        text = _TABLE_ROW_RE.sub(_table_line, text)

    # Structural markers -> gone (but keep the content they prefixed).
    text = _HR_RE.sub("\n", text)
    text = _HEADER_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _LIST_BULLET_RE.sub("", text)
    text = _ORDERED_RE.sub("", text)

    # Emphasis -> plain.
    text = _STRIKE_RE.sub(lambda m: m.group(1), text)
    # Apply emphasis stripping twice to catch nested ***bold italic***.
    for _ in range(2):
        text = _BOLD_ITALIC_RE.sub(lambda m: m.group(2), text)

    # Whitespace normalization.
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    return text.strip()


# --- WAV encoding ------------------------------------------------------------

def _float_to_wav_bytes(samples, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode a sequence of float samples in [-1, 1] as a 16-bit PCM mono WAV.

    Accepts a numpy array or a plain Python sequence. numpy is present whenever
    Kokoro is (it's a transitive dep), but we don't import it at module load so
    the cleaner stays dependency-free.
    """
    try:
        import numpy as np  # available alongside Kokoro
        arr = np.asarray(samples, dtype="float32").flatten()
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype("<i2").tobytes()
    except Exception:
        # Pure-stdlib fallback if numpy isn't around (e.g. a non-engine caller).
        clipped = [max(-1.0, min(1.0, float(s))) for s in samples]
        pcm = b"".join(struct.pack("<h", int(s * 32767)) for s in clipped)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def silence_wav(seconds: float = 0.25, sample_rate: int = SAMPLE_RATE) -> bytes:
    """A short stretch of silence, used as a safe non-empty TTS response."""
    n = int(seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


# --- The engine --------------------------------------------------------------

class KokoroTTS:
    """Thin, resident wrapper over Kokoro's ``KPipeline``.

    One pipeline per language code, loaded lazily on first use and cached. All
    methods that touch the engine raise :class:`TTSUnavailable` if Kokoro/torch
    isn't importable, so the caller never has to special-case the import itself.
    """

    def __init__(self, device: str = "auto"):
        self.device = device
        self._pipelines: dict[str, object] = {}
        self._lock = threading.Lock()

    def _get_pipeline(self, lang_code: str):
        with self._lock:
            pipe = self._pipelines.get(lang_code)
            if pipe is not None:
                return pipe
            try:
                from kokoro import KPipeline  # heavy: torch import happens here
            except Exception as e:  # ImportError or torch load failure
                raise TTSUnavailable(
                    "Kokoro is not installed. Install voice deps with "
                    "`pip install -r requirements-voice.txt`."
                ) from e
            try:
                pipe = KPipeline(lang_code=lang_code)
            except Exception as e:
                raise TTSUnavailable(f"Kokoro pipeline load failed: {e}") from e
            self._pipelines[lang_code] = pipe
            return pipe

    def warm(self, voice: str = DEFAULT_VOICE) -> None:
        """Pre-load the pipeline AND run a tiny synth so the first read isn't cold.

        Loading the pipeline is most of the cost, but Kokoro's first synthesis
        also does one-time graph/setup work — so we synthesize a throwaway word
        here to absorb that too. Best-effort: callers run this in a background
        thread and swallow :class:`TTSUnavailable` so a box without the engine
        still boots.
        """
        self._get_pipeline(lang_for_voice(voice))
        try:
            self.synthesize("ok", voice=voice)  # absorb first-inference cost
        except Exception:
            pass

    def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        read_code: bool = False,
    ) -> bytes:
        """Clean ``text`` (Markdown) and synthesize it to WAV bytes.

        Blocking — the endpoint dispatches this via ``asyncio.to_thread``. Returns
        a short silence WAV (never zero-length audio) when there's nothing
        speakable, matching the design's code-only-answer fallback.
        """
        speakable = markdown_to_speech(text, read_code=read_code)
        if not speakable.strip():
            return silence_wav()

        if voice not in _VOICE_IDS:
            voice = DEFAULT_VOICE
        speed = max(0.5, min(2.0, float(speed)))

        pipe = self._get_pipeline(lang_for_voice(voice))
        chunks = []
        try:
            # KPipeline yields (graphemes, phonemes, audio) per sentence/segment.
            for _gs, _ps, audio in pipe(speakable, voice=voice, speed=speed):
                if audio is not None:
                    chunks.append(audio)
        except Exception as e:
            raise TTSUnavailable(f"Kokoro synthesis failed: {e}") from e

        if not chunks:
            return silence_wav()

        try:
            import numpy as np
            joined = np.concatenate([np.asarray(c, dtype="float32").flatten() for c in chunks])
        except Exception:
            joined = [float(s) for c in chunks for s in c]
        return _float_to_wav_bytes(joined)

    def ready(self, voice: str = DEFAULT_VOICE) -> None:
        """Load the pipeline for ``voice`` (raises TTSUnavailable if absent).

        Called by the streaming endpoint *before* it starts the response, so an
        absent engine becomes a clean 503 instead of a half-open stream. Unlike
        warm(), it does not synthesize — it just ensures the pipeline is loaded.
        """
        self._get_pipeline(lang_for_voice(voice))

    def synth_stream(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        read_code: bool = False,
    ):
        """Yield length-prefixed WAV chunks, one per synthesized sentence.

        This is the streaming counterpart to :meth:`synthesize`. Kokoro's pipeline
        already yields audio per segment as it synthesizes, so we encode and emit
        each segment immediately instead of concatenating the whole answer first.
        The browser plays sentence one while sentence two is still synthesizing —
        time-to-first-audio becomes the cost of the FIRST sentence, not the whole
        response.

        Wire format (so the browser can parse chunks as they arrive):
            [4-byte big-endian length N][N bytes: a self-contained WAV]  ... repeat

        Generator + blocking — the endpoint wraps it in a StreamingResponse, which
        iterates it in a threadpool, so each yield flushes as its segment finishes.
        """
        speakable = markdown_to_speech(text, read_code=read_code)
        if not speakable.strip():
            wav = silence_wav()
            yield struct.pack(">I", len(wav)) + wav
            return

        if voice not in _VOICE_IDS:
            voice = DEFAULT_VOICE
        speed = max(0.5, min(2.0, float(speed)))
        pipe = self._get_pipeline(lang_for_voice(voice))

        emitted = False
        try:
            for _gs, _ps, audio in pipe(speakable, voice=voice, speed=speed):
                if audio is None:
                    continue
                try:
                    import numpy as np
                    arr = np.asarray(audio, dtype="float32").flatten()
                except Exception:
                    arr = audio
                wav = _float_to_wav_bytes(arr)
                emitted = True
                yield struct.pack(">I", len(wav)) + wav
        except Exception as e:
            # Mid-stream failure: if nothing went out yet, surface it so the route
            # can 503; otherwise the stream just ends where it got to.
            if not emitted:
                raise TTSUnavailable(f"Kokoro streaming synthesis failed: {e}") from e
            return

        if not emitted:
            wav = silence_wav()
            yield struct.pack(">I", len(wav)) + wav


# Module-level singleton, mirroring how nChat holds one Ollama client posture.
_engine: KokoroTTS | None = None
_engine_lock = threading.Lock()


def get_engine(device: str = "auto") -> KokoroTTS:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = KokoroTTS(device=device)
        return _engine


# --- Phase 0 isolation harness ----------------------------------------------

if __name__ == "__main__":
    import sys

    sample = (
        "# Quick check\n\n"
        "Configure **OSPF** on the `core-01` switch, then verify with this:\n\n"
        "```bash\nshow ip ospf neighbor\nshow ip route ospf\n```\n\n"
        "- Area 0 is the backbone\n"
        "- Cost is *inversely* proportional to bandwidth\n\n"
        "See [the docs](https://example.com/ospf) for the rest.\n"
    )
    print("=== markdown_to_speech (announce code, default) ===")
    print(repr(markdown_to_speech(sample)))
    print("\n=== markdown_to_speech (read_code=True) ===")
    print(repr(markdown_to_speech(sample, read_code=True)))
    print(f"\n=== voices: {len(list_voices())} ===")
    for v in list_voices():
        print(f"  {v['id']:14s} {v['label']}")

    if "--synth" in sys.argv:
        # Only runs if Kokoro is actually installed on this box.
        try:
            wav = get_engine().synthesize(sample)
            out = "/tmp/nchat_tts_selftest.wav"
            with open(out, "wb") as f:
                f.write(wav)
            print(f"\nWrote {len(wav)} bytes -> {out}")
        except TTSUnavailable as e:
            print(f"\n[engine unavailable] {e}")