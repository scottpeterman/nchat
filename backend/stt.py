"""nChat STT — local faster-whisper wrapper with networking-aware refinement.

Self-hosted speech-to-text for nChat. Caches one ``WhisperModel``, dispatches
blocking transcription to a threadpool from the caller, and returns the raw
transcript plus a refined version. STT drops text into the textarea; it never
auto-sends — review before send.

The value-add over a raw Whisper call is the refinement stage, lifted from
velocidictate:

* a **regex pass** (default, zero latency) that converts spoken punctuation
  ("comma", "new paragraph") to marks and repairs the handful of ways Whisper
  mangles protocol names ("oh SPF" -> OSPF), using a hand-built networking vocab
  pack; and
* an optional **local-LLM pass** ("ollama" mode) that runs a fixed networking
  correction prompt against the model already in front of nChat, for the gnarly
  dictation that the regex pass can't reach.

Design contract:

* **Model cached once, blocking inference dispatched to a thread, bytes in /
  text out.** Identical shape to ``tts.py``.
* **Lazy, guarded engine import.** Importing this module never requires
  faster-whisper or CTranslate2. If the runtime/weights are absent,
  :meth:`WhisperSTT.transcribe` raises :class:`STTUnavailable`.
* **``device="auto"`` with a CPU fallback.** Resolves to CPU int8 on a Mac with
  no CUDA; on a contended CUDA box, an OOM retries on CPU rather than crashing.
* **Domain is fixed (networking).** velocidictate's domain selector collapses —
  the networking prompt and vocab are baked in. The vocab pack is the durable
  asset and transfers verbatim into the LLM prompt regardless of refine backend.

The regex refinement, vocab pack, and prompt builder are stdlib-only and fully
testable without the engine (Phase 0 gate). Run ``python -m backend.stt`` for
the isolation harness.
"""
from __future__ import annotations

import re
import threading

import httpx


class STTUnavailable(RuntimeError):
    """Raised when the faster-whisper runtime or weights are not present."""


# --- Networking vocabulary pack ----------------------------------------------
#
# Hand-built knowledge of how Whisper mishears protocol names and CLI tokens.
# This is the durable asset: it's reused by both the regex pass (as direct
# substitutions) and the LLM pass (folded into the prompt).
#
# ONE source of truth. Whisper emits acronyms two ways — joined ("mtu") and as
# spaced single letters ("m t u") — and we generate the spaced-letter pattern
# from each canonical entry, so adding an acronym here fixes BOTH forms at once
# rather than hand-maintaining parallel lists.

# Canonical acronyms: lowercase letters-only key -> canonical casing.
_ACRONYMS: dict[str, str] = {
    "ospf": "OSPF", "bgp": "BGP", "vlan": "VLAN", "vxlan": "VXLAN",
    "vrf": "VRF", "svi": "SVI", "dhcp": "DHCP", "lacp": "LACP",
    "snmp": "SNMP", "ntp": "NTP", "tcp": "TCP", "udp": "UDP",
    "mpls": "MPLS", "ldp": "LDP", "rsvp": "RSVP", "eigrp": "EIGRP",
    "arp": "ARP", "nat": "NAT", "acl": "ACL", "qos": "QoS",
    "stp": "STP", "rstp": "RSTP", "mstp": "MSTP", "lag": "LAG",
    "sfp": "SFP", "poe": "PoE", "vpn": "VPN", "wan": "WAN",
    "lan": "LAN", "dns": "DNS", "ssh": "SSH", "tls": "TLS",
    "asn": "ASN", "mtu": "MTU", "cdp": "CDP", "lldp": "LLDP",
    "bfd": "BFD", "rib": "RIB", "fib": "FIB", "ecmp": "ECMP",
}

# Irregular forms that don't follow the generic spaced-letters rule: the
# homophone "oh"/"o" for the letter O, mixed-case canonicals (iBGP/eBGP), and
# acronyms whose canonical contains a hyphen (IS-IS, NX-OS). Applied BEFORE the
# generic pass so "i b g p" becomes iBGP rather than "i BGP".
_IRREGULAR: list[tuple[str, str]] = [
    (r"\boh\s*s\s*p\s*f\b", "OSPF"),
    (r"\bo\s*s\s*p\s*f\b", "OSPF"),
    (r"\bi\s+b\s*g\s*p\b", "iBGP"),
    (r"\be\s+b\s*g\s*p\b", "eBGP"),
    (r"\bi\s*s\s*i\s*s\b", "IS-IS"),
    (r"\bisis\b", "IS-IS"),
    (r"\bn\s*x\s*o\s*s\b", "NX-OS"),
    (r"\bnxos\b", "NX-OS"),
]

# Whole-token canonical-casing fixups that aren't simple letter acronyms:
# protocol-version names and vendor/OS names Whisper lowercases.
_TOKEN_FIXUPS: dict[str, str] = {
    "ipv4": "IPv4", "ipv6": "IPv6", "cli": "CLI", "api": "API",
    "junos": "Junos", "eos": "EOS", "ios": "IOS",
    "cisco": "Cisco", "arista": "Arista", "juniper": "Juniper",
}


def _spaced_pattern(key: str) -> str:
    """Build a regex matching an acronym whether joined or spaced into letters.

    'mtu' -> r'\\bm\\s*t\\s*u\\b', which matches 'mtu', 'm t u', 'M  T  U'.
    """
    return r"\b" + r"\s*".join(re.escape(ch) for ch in key) + r"\b"


# Pre-compile the generic acronym pass, longest keys first so a longer acronym
# isn't pre-empted by a shorter one sharing a prefix.
_ACRONYM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(_spaced_pattern(k), re.IGNORECASE), v)
    for k, v in sorted(_ACRONYMS.items(), key=lambda kv: len(kv[0]), reverse=True)
]

_DASH_JOIN_RE = re.compile(r"(?<=[\w])\s*-\s*(?=[\w])")

# Spoken-punctuation -> mark. Applied case-insensitively. Marks that attach to
# the preceding word (no leading space) are handled in _attach_punctuation.
_SPOKEN_PUNCT: list[tuple[str, str]] = [
    (r"\bnew\s+paragraph\b", "\n\n"),
    (r"\bnew\s+line\b", "\n"),
    (r"\bnewline\b", "\n"),
    (r"\bopen\s+paren(?:thesis)?\b", "("),
    (r"\bclose\s+paren(?:thesis)?\b", ")"),
    (r"\bopen\s+bracket\b", "["),
    (r"\bclose\s+bracket\b", "]"),
    (r"\bopen\s+brace\b", "{"),
    (r"\bclose\s+brace\b", "}"),
    (r"\bcomma\b", ","),
    (r"\bperiod\b", "."),
    (r"\bfull\s+stop\b", "."),
    (r"\bquestion\s+mark\b", "?"),
    (r"\bexclamation\s+(?:point|mark)\b", "!"),
    (r"\bcolon\b", ":"),
    (r"\bsemicolon\b", ";"),
    (r"\bhyphen\b", "-"),
    (r"\bdash\b", "-"),
    (r"\bslash\b", "/"),
    (r"\bforward\s+slash\b", "/"),
    (r"\bdot\b", "."),
]

_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:])")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_SPACE_NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")


def _apply_vocab(text: str) -> str:
    # Irregular forms first (oh-spf, iBGP/eBGP, IS-IS, NX-OS).
    for pat, repl in _IRREGULAR:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    # Generic acronyms (joined or spaced), longest first.
    for pat, repl in _ACRONYM_PATTERNS:
        text = pat.sub(repl, text)
    # Whole-token name fixups (IPv4, vendors, OS names).
    keys = sorted(_TOKEN_FIXUPS.keys(), key=len, reverse=True)
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE)
    text = pat.sub(lambda m: _TOKEN_FIXUPS.get(m.group(0).lower(), m.group(0)), text)
    return text


def _apply_spoken_punct(text: str) -> str:
    for pat, mark in _SPOKEN_PUNCT:
        text = re.sub(pat, mark, text, flags=re.IGNORECASE)
    return text


def _tidy(text: str) -> str:
    text = _DASH_JOIN_RE.sub("-", text)              # "core - 1" -> "core-1"
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)   # "word ," -> "word,"
    text = _MULTISPACE_RE.sub(" ", text)
    text = _SPACE_NEWLINE_RE.sub("\n", text)
    # Capitalize the first alpha character of the whole utterance.
    text = text.strip()
    for i, ch in enumerate(text):
        if ch.isalpha():
            text = text[:i] + ch.upper() + text[i + 1:]
            break
    return text


def refine_regex(raw: str) -> str:
    """Zero-latency refinement: spoken punctuation + networking vocab fixups.

    This is the daily-driver default. Whisper 'small' with VAD already gets a lot
    of network vocabulary right, and you review before sending, so this pass just
    cleans the obvious wins. Pure stdlib.
    """
    if not raw:
        return ""
    text = _apply_spoken_punct(raw)
    text = _apply_vocab(text)
    return _tidy(text)


# --- LLM refinement (optional) -----------------------------------------------

# Fixed networking prompt — the domain selector from velocidictate collapses to
# this one baked prompt. The vocab pack is folded in so the LLM knows the target
# spellings even for terms the regex pass didn't catch.
_VOCAB_HINT = ", ".join(sorted(
    set(_ACRONYMS.values())
    | set(_TOKEN_FIXUPS.values())
    | {"OSPF", "iBGP", "eBGP", "IS-IS", "NX-OS"}
))

REFINE_SYSTEM_PROMPT = (
    "You clean up dictated text from a network engineer. Fix spoken punctuation "
    "and correct mis-transcribed networking terms to their standard spelling and "
    "casing. Do NOT answer, explain, summarize, or add anything — return only the "
    "corrected text. Preserve the speaker's words and meaning exactly. "
    "Common terms to spell correctly: " + _VOCAB_HINT + "."
)


def refine_ollama(
    raw: str,
    ollama_base: str = "http://localhost:11434",
    model: str = "qwen3:30b-a3b",
    timeout: float = 8.0,
) -> str:
    """LLM refinement against the local Ollama already in front of nChat.

    Runs the fixed networking correction prompt. The rule is 'fast model, not
    dense large model' — an MoE like qwen3:30b-a3b (~3B active) keeps the
    round-trip short. Degrades to :func:`refine_regex` on *any* failure (Ollama
    down, timeout, bad response) so the transcript is never lost.
    """
    # Seed with the regex pass so even the LLM's input is already half-cleaned,
    # and so the fallback path returns something good.
    seeded = refine_regex(raw)
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(
                f"{ollama_base}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "think": False,
                    "keep_alive": "30m",
                    "messages": [
                        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                        {"role": "user", "content": seeded},
                    ],
                    "options": {"temperature": 0.0},
                },
            )
            r.raise_for_status()
            out = (r.json().get("message", {}) or {}).get("content", "").strip()
            # Guard against a chatty model that ignored the instruction.
            if out and len(out) <= max(40, len(seeded) * 3):
                return out
            return seeded
    except Exception:
        return seeded


def refine(raw: str, mode: str = "regex", **ollama_kwargs) -> str:
    """Dispatch refinement by mode: 'regex' (default), 'ollama', or 'none'."""
    if mode == "none":
        return raw
    if mode == "ollama":
        return refine_ollama(raw, **ollama_kwargs)
    return refine_regex(raw)


# --- The engine --------------------------------------------------------------

class WhisperSTT:
    """Thin, resident wrapper over faster-whisper's ``WhisperModel``.

    One model, loaded lazily on first use and cached. ``device='auto'`` resolves
    to CUDA float16 if available, else CPU int8; a CUDA OOM at load or decode
    retries on CPU. Engine methods raise :class:`STTUnavailable` if the runtime
    isn't importable.
    """

    def __init__(self, model_size: str = "small", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        self._resolved_device: str | None = None

    def _load(self):
        from faster_whisper import WhisperModel  # heavy: CTranslate2

        def _try(device: str, compute_type: str):
            return WhisperModel(self.model_size, device=device, compute_type=compute_type)

        if self.device == "cpu":
            self._resolved_device = "cpu"
            return _try("cpu", "int8")
        if self.device == "cuda":
            try:
                self._resolved_device = "cuda"
                return _try("cuda", "float16")
            except Exception:
                self._resolved_device = "cpu"
                return _try("cpu", "int8")
        # auto: prefer CUDA, fall back to CPU (the Apple-Silicon path lands here).
        try:
            self._resolved_device = "cuda"
            return _try("cuda", "float16")
        except Exception:
            self._resolved_device = "cpu"
            return _try("cpu", "int8")

    def _get_model(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                self._model = self._load()
            except Exception as e:
                raise STTUnavailable(
                    "faster-whisper is not installed. Install voice deps with "
                    "`pip install -r requirements-voice.txt`."
                ) from e
            return self._model

    def warm(self) -> None:
        """Pre-load the model so the first dictation isn't cold. Best-effort."""
        self._get_model()

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to raw text. Blocking — dispatch via a thread.

        VAD filtering is on (drops silence, tightens accuracy on short clips).
        A CUDA OOM at decode time retries once on CPU rather than failing.
        """
        model = self._get_model()
        try:
            segments, _info = model.transcribe(audio_path, vad_filter=True)
            return "".join(seg.text for seg in segments).strip()
        except Exception as e:
            # Decode-time OOM on a contended GPU: drop to CPU and retry once.
            if self._resolved_device == "cuda":
                try:
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                    self._resolved_device = "cpu"
                    segments, _info = self._model.transcribe(audio_path, vad_filter=True)
                    return "".join(seg.text for seg in segments).strip()
                except Exception as e2:
                    raise STTUnavailable(f"Whisper transcription failed: {e2}") from e2
            raise STTUnavailable(f"Whisper transcription failed: {e}") from e


_engine: WhisperSTT | None = None
_engine_lock = threading.Lock()


def get_engine(model_size: str = "small", device: str = "auto") -> WhisperSTT:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = WhisperSTT(model_size=model_size, device=device)
        return _engine


# --- Phase 0 isolation harness ----------------------------------------------

if __name__ == "__main__":
    import sys

    cases = [
        "configure oh SPF on core dash one comma then check B G P new paragraph "
        "verify the V LAN and the i B G P session period",
        "the m p l s core uses l d p comma not r s v p",
        "set the m t u to nine thousand on the lag comma enable q o s",
        "add a static route new line then commit",
    ]
    print("=== refine_regex ===")
    for c in cases:
        print(f"  raw : {c}")
        print(f"  out : {refine_regex(c)!r}\n")

    print("=== refine system prompt (baked) ===")
    print(REFINE_SYSTEM_PROMPT)

    if "--transcribe" in sys.argv and len(sys.argv) > 2:
        path = sys.argv[-1]
        try:
            raw = get_engine().transcribe(path)
            print(f"\nraw : {raw!r}")
            print(f"out : {refine_regex(raw)!r}")
        except STTUnavailable as e:
            print(f"\n[engine unavailable] {e}")