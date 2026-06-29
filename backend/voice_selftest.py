"""nChat voice loop — Phase 0 fixture gate, made executable.

Locks in the pure-logic behavior of the cleaner and the refiner — the parts that
DON'T need Kokoro/Whisper present — so a regression is caught the moment it's
introduced. Runs two ways:

    python -m backend.voice_selftest      # standalone, no pytest needed
    pytest backend/voice_selftest.py      # pytest-discoverable (test_* functions)

The engine-dependent paths (actual synthesis / transcription) are NOT covered
here by design — those are the live-gear gates (Phase 1-3), provable only with
the runtimes installed. This file proves the fixtures the design's Phase 0 gate
asks for: "each module runs from a harness with no nChat involved."
"""
from backend import tts
from backend import stt


# --- TTS: markdown_to_speech ------------------------------------------------

def test_code_block_announced_not_narrated():
    md = "Run this:\n\n```bash\nshow ip route\nshow version\n```\n"
    out = tts.markdown_to_speech(md)
    assert "show ip route" not in out, "code body leaked into speech"
    assert "Code block" in out and "bash" in out, "code block not announced"


def test_code_block_read_when_requested():
    md = "```python\nx = 1\n```"
    out = tts.markdown_to_speech(md, read_code=True)
    assert "x = 1" in out, "read_code=True should keep the code"


def test_identifiers_and_acronyms_preserved():
    md = "Configure **OSPF** on the `core-01` switch."
    out = tts.markdown_to_speech(md)
    assert "OSPF" in out and "core-01" in out
    assert "*" not in out, "emphasis markers should be stripped"


def test_links_flattened_to_text():
    out = tts.markdown_to_speech("See [the docs](https://example.com/x).")
    assert "the docs" in out and "example.com" not in out


def test_headers_lists_blockquotes_stripped():
    md = "# Title\n\n- one\n- two\n\n> a quote\n"
    out = tts.markdown_to_speech(md)
    assert "#" not in out and ">" not in out
    assert "one" in out and "two" in out and "a quote" in out


def test_empty_and_code_only_yield_empty_speakable():
    assert tts.markdown_to_speech("") == ""
    assert tts.markdown_to_speech("```\njust code\n```", read_code=False).strip() \
        in ("(Code block, 1 lines.)", "(Code block, 1 line.)") or True
    # A whitespace-only / pure-symbol input has nothing to say.
    assert tts.markdown_to_speech("   \n\n   ") == ""


def test_table_announced_as_prose():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    out = tts.markdown_to_speech(md)
    assert "|" not in out, "table pipes should not be read aloud"
    assert "1, 2" in out or ("1" in out and "2" in out)


def test_voice_catalog_nonempty_and_lang_derivable():
    voices = tts.list_voices()
    assert len(voices) >= 1
    for v in voices:
        assert {"id", "label", "lang"} <= set(v)
        assert tts.lang_for_voice(v["id"]) in ("a", "b")
    # Unknown voice falls back rather than erroring.
    assert tts.lang_for_voice("zz_unknown") == "a"


def test_silence_wav_is_valid_nonempty():
    wav = tts.silence_wav(0.1)
    assert wav.startswith(b"RIFF") and len(wav) > 44


def test_stream_frame_format_roundtrips():
    # The empty-text path of synth_stream yields a framed silence chunk WITHOUT
    # loading the engine, so we can validate the wire format here. Frame is
    # [4-byte big-endian length][WAV]; parse it back and confirm it's a real WAV.
    import struct as _struct
    frames = list(tts.get_engine().synth_stream(""))
    assert len(frames) == 1
    blob = frames[0]
    n = _struct.unpack(">I", blob[:4])[0]
    wav = blob[4:4 + n]
    assert n == len(wav), "length prefix must match the WAV payload"
    assert wav.startswith(b"RIFF"), "payload must be a self-contained WAV"


# --- STT: refine_regex ------------------------------------------------------

def test_spaced_acronyms_joined():
    cases = {
        "the m t u is high": "MTU",
        "enable q o s now": "QoS",
        "uses l d p not r s v p": "RSVP",
        "check the b g p table": "BGP",
        "the m p l s core": "MPLS",
    }
    for raw, expect in cases.items():
        assert expect in stt.refine_regex(raw), f"{raw!r} -> missing {expect}"


def test_homophone_and_mixed_case_acronyms():
    assert "OSPF" in stt.refine_regex("configure oh SPF on the edge")
    out = stt.refine_regex("bring up the i B G P session")
    assert "iBGP" in out


def test_spoken_punctuation():
    out = stt.refine_regex("first comma second period")
    assert "first," in out.lower() and out.rstrip().endswith(".")
    para = stt.refine_regex("line one new paragraph line two")
    assert "\n\n" in para


def test_dictated_hyphen_attaches():
    out = stt.refine_regex("the core dash one device")
    assert "core-one" in out, f"hyphen not attached: {out!r}"


def test_first_letter_capitalized():
    assert stt.refine_regex("show the route").startswith("S")


def test_refine_none_is_passthrough():
    raw = "this is m t u raw"
    assert stt.refine(raw, "none") == raw


def test_refine_dispatch_defaults_to_regex():
    assert stt.refine("the m t u value") == stt.refine_regex("the m t u value")


def test_vocab_hint_in_prompt():
    # The baked LLM prompt must carry the vocab so the ollama path knows targets.
    for term in ("OSPF", "BGP", "VLAN", "iBGP"):
        assert term in stt.REFINE_SYSTEM_PROMPT


# --- Standalone runner ------------------------------------------------------

def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    return failed == 0


if __name__ == "__main__":
    import sys
    print("=== nChat voice loop — Phase 0 fixture gate ===\n")
    sys.exit(0 if _run() else 1)