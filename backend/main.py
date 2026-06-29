"""nChat - FastAPI backend for local LLM chat via Ollama"""
import json
import os
import time
import httpx
import asyncio
import tempfile
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from . import database as db
from . import search as websearch

# Voice loop (TTS + STT). Imported guarded so nChat boots even if these files are
# absent — delete tts.py / stt.py and their routes and nChat is unchanged. The
# heavy engine import (torch / CTranslate2) is lazy *inside* these modules, so
# importing them here is cheap and never requires the ML runtimes to be present.
try:
    from . import tts as tts_mod
except Exception:
    tts_mod = None
try:
    from . import stt as stt_mod
except Exception:
    stt_mod = None

OLLAMA_BASE = "http://localhost:11434"
STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"

# --- Context budgeting + upload config ---
# Ollama silently caps context to num_ctx (historically 4096) unless told
# otherwise, so anything we inject past that limit is dropped before the model
# ever sees it. We size num_ctx per request from an estimate of the prompt.
DEFAULT_NUM_CTX = 8192               # floor — comfortably above Ollama's default
MAX_NUM_CTX = 32768                  # ceiling — your VRAM guard; tune to the GPU
RESPONSE_RESERVE_TOKENS = 4096       # headroom left for the model's own reply (code answers run long)
MAX_FILE_TOKENS = 8000               # per-file injection cap (truncate beyond)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # 5 MB hard limit per upload
KEEP_ALIVE = "30m"                   # keep the model resident between turns (avoids reload TTFT)

# --- Voice loop config ---
# All opt-in / environment-tunable so the default boot stays instant and lean.
VOICE_WARM = os.environ.get("NCHAT_VOICE_WARM", "0") == "1"   # pre-load engines at startup
TTS_CACHE_ENABLED = os.environ.get("NCHAT_TTS_CACHE", "0") == "1"  # opt-in disk cache for replays
TTS_CACHE_DIR = Path(__file__).parent / "tts_cache"
TTS_CACHE_MAX = int(os.environ.get("NCHAT_TTS_CACHE_MAX", "128"))  # LRU bound, by entry count
REFINE_MODEL = os.environ.get("NCHAT_REFINE_MODEL", "qwen3:30b-a3b")  # MoE: fast, not dense-large
WHISPER_SIZE = os.environ.get("NCHAT_WHISPER_SIZE", "small")
# STT latency levers (tune without code edits). For lower latency try:
#   NCHAT_WHISPER_SIZE=base.en  (English-only, faster + more accurate than 'small'
#   for English; 'tiny.en' is fastest), NCHAT_WHISPER_BEAM=1, NCHAT_VOICE_WARM=1.
WHISPER_BEAM = int(os.environ.get("NCHAT_WHISPER_BEAM", "1"))   # 1 = greedy (fastest)
WHISPER_VAD = os.environ.get("NCHAT_WHISPER_VAD", "1") == "1"   # trim silence before decode


def _stt_engine():
    """The shared Whisper engine, configured from the environment."""
    return stt_mod.get_engine(
        model_size=WHISPER_SIZE, beam_size=WHISPER_BEAM, vad_filter=WHISPER_VAD,
    )

# Clear-text only. Extensionless files (e.g. running-configs) are allowed and
# fall back to the binary-detection gate below.
ALLOWED_EXTENSIONS = {
    ".txt", ".text", ".md", ".markdown", ".log",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".sh", ".sql",
    ".json", ".csv", ".tsv", ".xml", ".html", ".css",
    ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg", ".env", ".j2",
}


def estimate_tokens(text: str) -> int:
    """Rough heuristic, deliberately conservative: ~3.5 chars/token. Prose is
    closer to 4, but code/config is denser, and we'd rather over-estimate the
    prompt (sizing num_ctx a little large) than under-estimate it and let
    generation run into the context ceiling. (len*2)//7 == len/3.5."""
    return max(1, (len(text) * 2) // 7)


def round_up(n: int, step: int = 1024) -> int:
    return ((n + step - 1) // step) * step


def is_probably_text(raw: bytes) -> bool:
    """Reject binary uploads. A null byte is the cheapest reliable signal."""
    return b"\x00" not in raw


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    if estimate_tokens(text) <= max_tokens:
        return text, False
    return text[: max_tokens * 4], True


def inject_files(content: str, files: list[dict]) -> str:
    """Append attached-file context after the user's prose so the question
    still reads first. Each file is capped and marked if truncated."""
    blocks = [content]
    for f in files:
        body, truncated = truncate_to_tokens(f["content"], MAX_FILE_TOKENS)
        note = ""
        if truncated:
            note = (f"\n\n[... truncated: showing ~{MAX_FILE_TOKENS} "
                    f"of ~{f['est_tokens']} estimated tokens ...]")
        blocks.append(
            f"\n\n--- Attached file: {f['filename']} ---\n{body}{note}"
            f"\n--- End of {f['filename']} ---"
        )
    return "".join(blocks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Pre-warm voice engines in the background if asked. Best-effort: a box
    # without the ML runtimes (or the weights) just logs and moves on — the app
    # boots regardless, and the engines load lazily on first use instead.
    if VOICE_WARM:
        def _warm():
            if tts_mod is not None:
                try:
                    tts_mod.get_engine().warm()
                    print("[voice] TTS warmed")
                except Exception as e:
                    print(f"[voice] TTS warm skipped: {e}")
            if stt_mod is not None:
                try:
                    _stt_engine().warm()
                    print("[voice] STT warmed")
                except Exception as e:
                    print(f"[voice] STT warm skipped: {e}")
        threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="nChat", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    model: str = "qwen2.5-coder:32b"
    system_prompt: str = ""
    think: bool = False          # ask thinking-capable models to stream their reasoning
    file_ids: list[str] = []
    num_ctx: int | None = None   # optional manual override; clamped to MAX_NUM_CTX
    web_search: bool = False     # ground this turn in live web results
    search_results: int = 5      # how many results to inject (1–10)


class ConversationCreate(BaseModel):
    model: str = ""
    title: str = "New Conversation"
    system_prompt: str = ""


class ConversationUpdate(BaseModel):
    title: str | None = None
    model: str | None = None
    system_prompt: str | None = None


class PromptCreate(BaseModel):
    name: str
    content: str


# --- Ollama proxy ---

async def check_ollama() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
            return r.status_code == 200
    except Exception:
        return False


@app.get("/api/health")
async def health():
    ollama_ok = await check_ollama()
    return {"status": "ok", "ollama": ollama_ok}


@app.get("/api/models")
async def list_models():
    """List available Ollama models."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags", timeout=10.0)
            r.raise_for_status()
            data = r.json()
            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m["name"],
                    "size": m.get("size", 0),
                    "parameter_size": m.get("details", {}).get("parameter_size", ""),
                    "quantization": m.get("details", {}).get("quantization_level", ""),
                    "modified_at": m.get("modified_at", ""),
                })
            return {"models": models}
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama is not running. Start it with: ollama serve")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream a chat response from Ollama."""
    # Get or create conversation
    if req.conversation_id:
        conv = db.get_conversation(req.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = db.create_conversation(model=req.model, system_prompt=req.system_prompt)

    conv_id = conv["id"]

    # Save user message, then bind any attached files to it
    user_msg = db.add_message(conv_id, "user", req.message)
    db.attach_files_to_message(user_msg["id"], req.file_ids, conversation_id=conv_id)
    db.auto_title(conv_id)

    # Keep model + system prompt on the conversation in sync with this request
    convo_updates = {}
    if req.model and conv.get("model") != req.model:
        convo_updates["model"] = req.model
    if conv.get("system_prompt", "") != req.system_prompt:
        convo_updates["system_prompt"] = req.system_prompt
    if convo_updates:
        db.update_conversation(conv_id, **convo_updates)

    # Build message history for Ollama, injecting any files attached per message
    history = db.get_messages(conv_id)
    ollama_messages = []

    if req.system_prompt:
        ollama_messages.append({"role": "system", "content": req.system_prompt})

    for msg in history:
        content = msg["content"]
        attached = db.get_files_for_message(msg["id"])
        if attached:
            content = inject_files(content, attached)
        ollama_messages.append({"role": msg["role"], "content": content})

    # --- Optional web search grounding (this turn only) ---
    # Injected before prompt_tokens is summed, so num_ctx grows to fit the
    # results and they aren't silently truncated by Ollama's default ceiling.
    search_sources: list[dict] = []
    search_notice = ""
    if req.web_search:
        provider = websearch.get_provider()
        if provider is None:
            search_notice = "Web search is not configured on the server."
        else:
            q = websearch.shape_query(req.message)
            n = min(max(req.search_results, 1), 10)
            try:
                results = await provider.search(q, max_results=n)
                if results:
                    block = websearch.format_results_block(q, results)
                    # Inject into the latest user turn (mirrors inject_files).
                    for m in reversed(ollama_messages):
                        if m["role"] == "user":
                            m["content"] = f"{m['content']}\n\n{block}"
                            break
                    # Add the grounding/abstention contract to the system msg.
                    if ollama_messages and ollama_messages[0]["role"] == "system":
                        ollama_messages[0]["content"] += "\n\n" + websearch.SEARCH_SYSTEM_ADDENDUM
                    else:
                        ollama_messages.insert(0, {
                            "role": "system",
                            "content": websearch.SEARCH_SYSTEM_ADDENDUM,
                        })
                    search_sources = [r.as_dict() for r in results]
                else:
                    search_notice = "Web search returned no results."
            except websearch.SearchError as e:
                search_notice = f"Web search unavailable: {e}"

        # Asked to search but got nothing usable: tell the model not to pretend
        # it did (ABSENT/UNREACHABLE, not silently unsourced).
        if not search_sources:
            if ollama_messages and ollama_messages[0]["role"] == "system":
                ollama_messages[0]["content"] += "\n\n" + websearch.SEARCH_UNAVAILABLE_ADDENDUM
            else:
                ollama_messages.insert(0, {
                    "role": "system",
                    "content": websearch.SEARCH_UNAVAILABLE_ADDENDUM,
                })

    # Size num_ctx from the actual prompt so injected files aren't silently
    # truncated by Ollama. Manual override wins but is clamped to the ceiling.
    prompt_tokens = sum(estimate_tokens(m["content"]) for m in ollama_messages)
    # Reasoning traces can be long and share the output budget with the answer,
    # so leave extra room when thinking is enabled.
    reserve = 8192 if req.think else RESPONSE_RESERVE_TOKENS
    if req.num_ctx:
        num_ctx = min(req.num_ctx, MAX_NUM_CTX)
    else:
        needed = prompt_tokens + reserve
        num_ctx = min(MAX_NUM_CTX, max(DEFAULT_NUM_CTX, round_up(needed)))

    async def generate():
        full_response = ""
        full_thinking = ""
        tokens_eval = 0
        tokens_prompt = 0
        duration_ms = 0
        saved = False

        def persist():
            # Save whatever we have — including partial output if the client
            # aborted mid-stream. Guarded so the normal + cancel paths don't double-save.
            nonlocal saved
            if (full_response or full_thinking) and not saved:
                db.add_message(
                    conv_id, "assistant", full_response,
                    model=req.model, tokens_eval=tokens_eval,
                    tokens_prompt=tokens_prompt, duration_ms=duration_ms,
                    thinking=full_thinking,
                    sources=json.dumps(search_sources) if search_sources else "",
                )
                saved = True

        # Send conversation_id + the chosen context size first
        yield f"data: {json.dumps({'conversation_id': conv_id, 'type': 'meta', 'num_ctx': num_ctx, 'prompt_tokens_est': prompt_tokens})}\n\n"

        if req.web_search:
            yield f"data: {json.dumps({'type': 'search', 'sources': search_sources, 'notice': search_notice})}\n\n"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE}/api/chat",
                    json={
                        "model": req.model,
                        "messages": ollama_messages,
                        "stream": True,
                        "think": req.think,
                        "keep_alive": KEEP_ALIVE,
                        "options": {"num_ctx": num_ctx},
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "error" in chunk:
                            yield f"data: {json.dumps({'error': chunk['error'], 'type': 'error'})}\n\n"
                            return

                        msg = chunk.get("message", {})

                        thinking = msg.get("thinking", "")
                        if thinking:
                            full_thinking += thinking
                            yield f"data: {json.dumps({'thinking': thinking, 'type': 'thinking'})}\n\n"

                        token = msg.get("content", "")
                        if token:
                            full_response += token
                            yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"

                        if chunk.get("done"):
                            tokens_eval = chunk.get("eval_count", 0)
                            tokens_prompt = chunk.get("prompt_eval_count", 0)
                            total_ns = chunk.get("total_duration", 0)
                            duration_ms = int(total_ns / 1_000_000) if total_ns else 0
                            eval_ns = chunk.get("eval_duration", 0)
                            tok_per_sec = (tokens_eval / (eval_ns / 1e9)) if eval_ns > 0 else 0

                            yield f"data: {json.dumps({'type': 'done', 'tokens_eval': tokens_eval, 'tokens_prompt': tokens_prompt, 'duration_ms': duration_ms, 'tokens_per_sec': round(tok_per_sec, 1), 'done_reason': chunk.get('done_reason', '')})}\n\n"

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to Ollama. Is it running?', 'type': 'error'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"
            return
        finally:
            # Runs on normal completion AND on client disconnect (CancelledError),
            # at which point the httpx context managers above have closed the
            # connection to Ollama, halting generation server-side.
            persist()

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- File upload ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...),
                      conversation_id: str | None = Form(None)):
    """Accept a clear-text file and return a file_id to pass in /api/chat.

    Gates, in order: extension allow-list (extensionless is permitted),
    size limit, non-empty, binary rejection, strict UTF-8 decode.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"File type '{ext}' not allowed. Clear-text/config/code files only.",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw)} bytes). Limit is {MAX_UPLOAD_BYTES} bytes.",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if not is_probably_text(raw):
        raise HTTPException(
            status_code=415,
            detail="File appears to be binary. Only clear-text files are supported.",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="File is not valid UTF-8 text.")

    rec = db.create_file(
        filename=file.filename or "upload.txt",
        content=text,
        size_bytes=len(raw),
        est_tokens=estimate_tokens(text),
        conversation_id=conversation_id,
    )
    return rec


# --- Conversation endpoints ---

@app.get("/api/conversations")
async def list_conversations(limit: int = 50, offset: int = 0):
    return {"conversations": db.list_conversations(limit, offset)}


@app.post("/api/conversations")
async def create_conversation(req: ConversationCreate):
    return db.create_conversation(model=req.model, title=req.title,
                                  system_prompt=req.system_prompt)


# --- Prompt / persona endpoints ---

@app.get("/api/prompts")
async def list_prompts():
    return {"prompts": db.list_prompts()}


@app.post("/api/prompts")
async def create_prompt(req: PromptCreate):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Preset name is required.")
    return db.create_prompt(name=name, content=req.content)


@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    if not db.delete_prompt(prompt_id):
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"deleted": True}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.put("/api/conversations/{conv_id}")
async def update_conversation(conv_id: str, req: ConversationUpdate):
    updates = req.model_dump(exclude_none=True)
    conv = db.update_conversation(conv_id, **updates)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if not db.delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


@app.get("/api/conversations/{conv_id}/messages")
async def get_messages(conv_id: str):
    conv = db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = db.get_messages(conv_id)
    for m in messages:
        m["files"] = db.get_file_meta_for_message(m["id"])
    return {"messages": messages}


# --- Voice loop endpoints (TTS + STT) ---
#
# Both degrade rather than fail: a missing engine surfaces a clear 503 and the
# rest of nChat keeps working. Audio is ephemeral — STT decodes from a temp file
# that's deleted immediately; TTS holds WAV bytes in memory just long enough to
# return them. Only the opt-in TTS cache persists audio, and only of text the
# database already holds.

class TTSRequest(BaseModel):
    text: str                    # raw Markdown; the backend strips it to prose
    voice: str = "af_heart"
    speed: float = 1.0
    read_code: bool = False      # default: announce code blocks, don't narrate them


def _tts_cache_path(speakable_key: str) -> Path:
    import hashlib
    h = hashlib.sha256(speakable_key.encode("utf-8")).hexdigest()[:32]
    return TTS_CACHE_DIR / f"{h}.wav"


def _tts_cache_get(key: str) -> bytes | None:
    if not TTS_CACHE_ENABLED:
        return None
    p = _tts_cache_path(key)
    if p.exists():
        try:
            data = p.read_bytes()
            p.touch()  # bump mtime for LRU
            return data
        except Exception:
            return None
    return None


def _tts_cache_put(key: str, wav: bytes) -> None:
    if not TTS_CACHE_ENABLED:
        return
    try:
        TTS_CACHE_DIR.mkdir(exist_ok=True)
        _tts_cache_path(key).write_bytes(wav)
        # Bounded LRU by count: evict oldest by mtime when over the cap.
        entries = sorted(TTS_CACHE_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime)
        for stale in entries[:-TTS_CACHE_MAX] if len(entries) > TTS_CACHE_MAX else []:
            stale.unlink(missing_ok=True)
    except Exception:
        pass  # cache is disposable by design; never let it break a read


@app.get("/api/tts/voices")
async def tts_voices():
    """Voice catalog for the dropdown. Available even without the engine loaded."""
    if tts_mod is None:
        return {"voices": []}
    return {"voices": tts_mod.list_voices()}


@app.post("/api/tts")
async def tts(req: TTSRequest):
    """Synthesize an assistant message (Markdown) to WAV and return audio/wav."""
    if tts_mod is None:
        raise HTTPException(status_code=503, detail="TTS module not available.")

    # Cache key is the *speakable* text (post-strip) + voice/speed/read_code, so
    # cosmetic Markdown changes that strip to the same prose still hit cache.
    speakable = tts_mod.markdown_to_speech(req.text, read_code=req.read_code)
    cache_key = f"{speakable}|{req.voice}|{req.speed}|{req.read_code}"

    cached = _tts_cache_get(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="audio/wav")

    try:
        wav = await asyncio.to_thread(
            tts_mod.get_engine().synthesize,
            req.text, req.voice, req.speed, req.read_code,
        )
    except tts_mod.TTSUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    _tts_cache_put(cache_key, wav)
    return Response(content=wav, media_type="audio/wav")


@app.post("/api/tts/stream")
async def tts_stream(req: TTSRequest):
    """Stream an assistant message as per-sentence WAV chunks (low TTFA).

    Returns a length-prefixed binary stream — `[4-byte BE length][WAV]...` — that
    the browser plays back-to-back while later sentences are still synthesizing.
    The engine pipeline is loaded *before* the response starts, so an absent
    engine is a clean 503 rather than a half-open stream. Falls back: the client
    uses the blob `/api/tts` if streaming isn't supported or errors early.
    """
    if tts_mod is None:
        raise HTTPException(status_code=503, detail="TTS module not available.")

    engine = tts_mod.get_engine()
    try:
        # Load the pipeline up front (raises if the runtime/weights are absent).
        await asyncio.to_thread(engine.ready, req.voice)
    except tts_mod.TTSUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    gen = engine.synth_stream(req.text, req.voice, req.speed, req.read_code)
    return StreamingResponse(
        gen,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/stt")
async def stt(file: UploadFile = File(...), refine: str = Form("regex")):
    """Transcribe an uploaded audio clip and return {raw, text}.

    `refine`: 'regex' (default, zero-latency vocab + punctuation), 'ollama'
    (LLM correction against the local model), or 'none' (raw Whisper). The text
    is returned for review — STT never auto-sends.
    """
    if stt_mod is None:
        raise HTTPException(status_code=503, detail="STT module not available.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    # Audio is ephemeral: write to a temp file only long enough for Whisper to
    # decode it (PyAV/CTranslate2 handle webm/opus), then delete.
    suffix = Path(file.filename or "clip.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(raw_bytes)
        tmp.flush()
        tmp.close()
        t0 = time.perf_counter()
        try:
            raw_text = await asyncio.to_thread(_stt_engine().transcribe, tmp.name)
        except stt_mod.STTUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
        t1 = time.perf_counter()

        if refine == "ollama":
            # LLM refinement reuses the Ollama already in front of nChat; it
            # degrades to the regex pass internally on any failure.
            text = await asyncio.to_thread(
                stt_mod.refine_ollama, raw_text, OLLAMA_BASE, REFINE_MODEL
            )
        else:
            text = stt_mod.refine(raw_text, refine)
        t2 = time.perf_counter()

        # Per-stage timing so latency can be tuned with data, not vibes. The
        # first call after a cold start carries model+VAD load here — watch the
        # delta between the first dictation and steady state.
        timings = {
            "transcribe_ms": round((t1 - t0) * 1000),
            "refine_ms": round((t2 - t1) * 1000),
            "audio_kb": round(len(raw_bytes) / 1024, 1),
        }
        return {"raw": raw_text, "text": text, "timings": timings}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# --- Static file serving (production) ---

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")