# nChat Voice Loop

Local text-to-speech (read responses aloud) and speech-to-text (dictate prompts),
both running on the same box as Ollama. **No audio, transcript, or generated text
crosses a network boundary you don't own.**

The two halves are independent: TTS alone gives you a "read this aloud" button;
STT alone gives you dictation into the input box. Together they close the loop —
speak the prompt, hear the answer.

This is a bolt-on. The chat spine — `/api/chat`, the SQLite schema, SSE streaming
— is untouched. Delete `backend/tts.py`, `backend/stt.py`, and their three routes
and nChat is exactly what it was.

---

## Install

The voice engines are the **one heavyweight exception** to nChat's lean stack
(torch via Kokoro, CTranslate2 via faster-whisper), so they're kept out of the
core `requirements.txt`. Install them only if you want voice:

```bash
source venv/bin/activate
pip install -r requirements-voice.txt
```

On Apple Silicon both ship native arm64 wheels — clean `pip install`, no source
builds. Weights download on first use and cache locally thereafter.

Then build the frontend and run as usual:

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8400
```

If you skip the voice install, nChat still runs — the Read/Mic buttons surface a
clear "not installed" message and everything else works (degrade, never fail).

---

## Use

- **Read aloud** — the `▶ Read` button in an assistant message's action row.
  Playback streams sentence by sentence, so it starts speaking almost immediately
  even on a long answer. Code blocks are *announced* ("Code block in bash, 3
  lines."), not narrated. Only one message reads at a time; click again to stop.
- **Dictate** — the 🎙 button beside the attach (`+`) button. Hold-to-talk:
  click to start, click to stop, transcript drops into the textarea. **It does
  not auto-send** — review and edit before you send.
- **Voice** — pick from the dropdown in the top bar (catalog served even before
  the engine loads).

### Microphone needs a secure context

`getUserMedia` requires HTTPS or `localhost` — a plain-HTTP LAN IP won't work
(the same secure-context rule the clipboard fallback already handles). Options:

1. Run the browser on the same box (localhost) — nothing to configure.
2. Front it with a TLS reverse proxy (Caddy with a local CA, or self-signed) for
   LAN access from a laptop. Pair with a bearer token, the same streamable-HTTP
   pattern mcpssh uses.
3. Browser origin flags — works but brittle; avoid.

---

## Configuration (environment variables)

| Variable | Default | Effect |
|----------|---------|--------|
| `NCHAT_VOICE_WARM` | `0` | `1` pre-loads both engines at startup (full decode path, background) so the first action isn't cold. Default lazy-loads on first use. |
| `NCHAT_WHISPER_SIZE` | `small` | faster-whisper model. `base.en` recommended for speed (faster *and* more accurate for English than `small`); `tiny.en` fastest. |
| `NCHAT_WHISPER_BEAM` | `1` | decode beam width; `1` = greedy (fastest, ~1.5–2× faster than the library default of 5). |
| `NCHAT_WHISPER_VAD` | `1` | `1` trims silence before decode (faster + more accurate on clips with dead air). |
| `NCHAT_REFINE_MODEL` | `qwen3:30b-a3b` | Ollama model for `ollama`-mode STT refinement. Rule: **fast model, not dense-large** — an MoE's ~3B active path keeps the round-trip short. |
| `NCHAT_TTS_CACHE` | `0` | `1` enables a bounded on-disk cache of synthesized WAVs (whole-WAV path only), so replays are instant. Disposable by design. |
| `NCHAT_TTS_CACHE_MAX` | `128` | LRU bound (entry count) for the TTS cache. |

**Recommended low-latency STT:** `NCHAT_VOICE_WARM=1 NCHAT_WHISPER_SIZE=base.en
NCHAT_WHISPER_BEAM=1` — takes steady-state transcription well under a second. The
`/api/stt` response carries per-stage `timings` (logged to the browser console)
so you can see where the time goes; the first call after a cold start carries
model+VAD load, and the delta to steady state tells you whether to warm.

Audio is ephemeral by default: STT decodes from a temp file that's deleted
immediately; TTS holds WAV bytes in memory just long enough to return them. The
optional cache is the only thing that persists audio, and only of text the
database already holds.

---

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| POST | `/api/tts/stream` | `{text, voice, speed, read_code}` | length-prefixed per-sentence WAV stream (near-instant first audio) |
| POST | `/api/tts` | `{text, voice, speed, read_code}` | `audio/wav` (whole-WAV; the fallback + cacheable form) |
| GET | `/api/tts/voices` | — | `{voices: [...]}` |
| POST | `/api/stt` | multipart: `file`, `refine` | `{raw, text, timings}` |

Read-aloud uses `/api/tts/stream` by default — the browser plays each sentence as
it's synthesized, so playback starts almost immediately even on a long answer. It
falls back to the whole-WAV `/api/tts` if the browser lacks Web Audio or the
stream errors before any audio plays.

`refine` is one of `regex` (default, zero-latency vocab + punctuation), `ollama`
(LLM correction against the local model; degrades to `regex` on any failure), or
`none` (raw Whisper).

---

## The placement decision (Apple Silicon)

Three models share the machine: the Ollama chat model (large), Whisper (small),
Kokoro (small). On a Mac there's no separate VRAM pool — all draw from unified
memory — so the question is total RAM, not headroom on a card. The voice models
add ~1 GB on top of the LLM, a rounding error against a 30B at q8.

The placement is settled: **chat model on Metal, voice models on CPU.** This is
*better* than a discrete-GPU box, not a compromise — the LLM runs on the GPU/Metal
cores while the voice models run on the CPU cores (different silicon), so they
overlap instead of fighting for one device. `device="auto"` in both modules
resolves to CPU for the voice models on a Mac with no CUDA, which is exactly right.

(A Linux/CUDA deployment reintroduces the original tradeoff — all three compete
for one GPU's VRAM. There the rule is LLM on GPU, voice on CPU when VRAM is
contended; the `device="auto"` → CPU-on-OOM fallback in `stt.py` keeps it
degrading rather than crashing.)

---

## Validation status

Built and de-risked in the spirit of the UglyFruit transition checklist —
fixture-proven first, then wired, then validated under real load. All proven on an
M-series Mac.

- **Modules in isolation.** `python -m backend.voice_selftest` runs the pure-logic
  fixture gate — the `markdown_to_speech` cleaner, the regex refiner and vocab
  pack, and the streaming frame format — with no engine and no nChat. `python -m
  backend.tts` and `python -m backend.stt` are the per-module harnesses.
- **Degrade, never fail.** App boots with no torch/CTranslate2; the voices
  endpoint serves the catalog regardless; TTS/STT return clean 503s; the spine is
  untouched and the voice routes aren't shadowed by the SPA catch-all.
- **TTS, localhost.** Responses read aloud with code blocks announced, not
  narrated; streaming starts speaking on the first sentence (near-instant); the
  Read button appears only after generation completes.
- **STT, localhost.** Mic dictation captures on localhost (secure-context prompt
  and all), transcribes sub-second with the recommended settings, and lands
  refined text in the textarea without auto-sending.
- **Coresidency.** The 30B MoE (30.3 GB), Whisper, and Kokoro run together; the
  full loop closes both directions with no contention felt — the live-gear gate,
  watching memory pressure rather than VRAM OOM.

---

## What's not here yet

- **Speed / read-code UI controls** — both are wired through the backend (`speed`,
  `read_code` on the TTS endpoints) and the voice settings module; only the voice
  dropdown is surfaced in the top bar so far. Surfacing the rest is a few lines.
- **Per-conversation voice/speed persistence** — voice is a global selection in
  the top bar today; storing it per-conversation in SQLite is the open call.
- **Auto-read** — button-triggered only by default; auto-reading every response is
  a setting, not a default (it spends CPU on every turn).
- **Hands-free mode** — the next natural direction now that both halves are fast:
  continuous VAD-gated capture, auto-send on a natural pause, auto-read with
  barge-in. A real build, not a knob, and deliberately off-by-default because it
  relaxes the review-before-send tenet. See §15 of the design doc.

## License

The voice backend is MIT-clean alongside nChat. Kokoro and its G2P are Apache-2.0;
faster-whisper and the Whisper weights are MIT. velocidictate's GPLv3 came from
PyQt6, which nChat doesn't use — so the lifted parts (Whisper wrapper, prompts,
vocab) shed the GPL trigger entirely. espeak-ng (an optional Kokoro fallback) is
GPLv3 but a separate runtime binary invoked as its own process — no license bleed.