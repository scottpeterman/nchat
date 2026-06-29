// nChat voice loop — shared browser-side plumbing for TTS playback and STT capture.
//
// One module owns the single shared Audio() object so only one message ever
// reads at a time, and exposes a tiny pub/sub so each ReadButton can reflect
// whether *it* is the one playing without threading state through props (which
// would defeat MessageBubble's memoization). STT capture lives here too so both
// halves of the loop share one mental model, mirroring the backend symmetry.

const API_BASE = '/api'

// --- Voice settings (module-level; set from the top-bar selector) ------------

let settings = { voice: 'af_heart', speed: 1.0, readCode: false }

export function setVoiceSettings(partial) {
  settings = { ...settings, ...partial }
}
export function getVoiceSettings() {
  return settings
}

export async function fetchVoices() {
  try {
    const r = await fetch(`${API_BASE}/tts/voices`)
    if (!r.ok) return []
    const d = await r.json()
    return d.voices || []
  } catch {
    return []
  }
}

// --- Shared playback controller ---------------------------------------------
//
// Two playback paths share one controller and one "only one message at a time"
// rule:
//   - STREAMING (default): /api/tts/stream sends per-sentence WAV chunks; we
//     decode each as it arrives and schedule them gaplessly on a Web Audio
//     timeline, so playback starts on sentence one while the rest synthesizes.
//   - BLOB (fallback): /api/tts returns one WAV; played via a single Audio().
//     Used when Web Audio/streaming isn't available, or if streaming errs before
//     any audio plays.

let audio = null               // blob-path <audio>
let lastUrl = null
let activeId = null            // id of the message currently loading/playing
let activeState = 'idle'       // 'idle' | 'loading' | 'playing'
const listeners = new Set()

// Streaming-path state.
let audioCtx = null
let streamAbort = null         // AbortController for the in-flight stream fetch
let scheduledSources = []      // BufferSources scheduled on the timeline
let idleTimer = null

function emit() {
  for (const fn of listeners) fn(activeId, activeState)
}

export function subscribe(fn) {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

function streamingSupported() {
  return typeof window !== 'undefined' &&
    (window.AudioContext || window.webkitAudioContext) &&
    typeof ReadableStream !== 'undefined'
}

function getCtx() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext
    audioCtx = new Ctx()
  }
  return audioCtx
}

function ensureAudio() {
  if (!audio) {
    audio = new Audio()
    audio.onended = () => { reset() }
    audio.onerror = () => { reset() }
  }
  return audio
}

function reset() {
  // Tear down the blob path.
  if (audio) { try { audio.pause() } catch {} }
  if (lastUrl) { try { URL.revokeObjectURL(lastUrl) } catch {}; lastUrl = null }
  // Tear down the streaming path.
  if (streamAbort) { try { streamAbort.abort() } catch {}; streamAbort = null }
  for (const s of scheduledSources) { try { s.stop() } catch {} }
  scheduledSources = []
  if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }

  activeId = null
  activeState = 'idle'
  emit()
}

export function stopSpeaking() {
  reset()
}

// Toggle reading for a message: if it's already active, stop; otherwise stop
// whatever's playing and read this. Throws on failure so the caller can toast.
export async function toggleSpeak(id, markdown) {
  if (activeId === id && (activeState === 'playing' || activeState === 'loading')) {
    reset()
    return
  }
  reset() // one at a time

  activeId = id
  activeState = 'loading'
  emit()

  if (streamingSupported()) {
    try {
      await streamSpeak(id, markdown)
      return
    } catch (err) {
      // If streaming failed before any audio played, fall back to the blob path.
      // (If we'd already started playing, reset() ran and we just surface it.)
      if (activeId !== id) return
      if (err && err.__played) { reset(); throw new Error(err.message) }
      // else fall through to blob
    }
  }
  await blobSpeak(id, markdown)
}

// Streaming playback: parse [4-byte BE length][WAV] frames, decode + schedule.
async function streamSpeak(id, markdown) {
  const ctx = getCtx()
  try { await ctx.resume() } catch {} // the Read click is the user gesture

  const ac = new AbortController()
  streamAbort = ac

  let r
  try {
    r = await fetch(`${API_BASE}/tts/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: markdown, voice: settings.voice,
        speed: settings.speed, read_code: settings.readCode,
      }),
      signal: ac.signal,
    })
  } catch (err) {
    throw new Error('Could not reach the TTS endpoint.')
  }
  if (!r.ok || !r.body) {
    const e = await r.json().catch(() => ({ detail: `TTS failed (${r.status})` }))
    throw new Error(e.detail || 'TTS failed')
  }

  const reader = r.body.getReader()
  let buf = new Uint8Array(0)
  let nextStart = 0
  let played = false

  const concat = (a, b) => {
    const out = new Uint8Array(a.length + b.length)
    out.set(a, 0); out.set(b, a.length)
    return out
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    if (activeId !== id) { try { reader.cancel() } catch {}; return }
    buf = concat(buf, value)

    // Drain every complete frame currently in the buffer.
    while (buf.length >= 4) {
      const view = new DataView(buf.buffer, buf.byteOffset, 4)
      const len = view.getUint32(0, false) // big-endian
      if (buf.length < 4 + len) break
      const wavBytes = buf.slice(4, 4 + len)
      buf = buf.slice(4 + len)

      let audioBuf
      try {
        audioBuf = await ctx.decodeAudioData(wavBytes.buffer.slice(0))
      } catch {
        continue // skip an undecodable chunk rather than killing playback
      }
      if (activeId !== id) return

      const src = ctx.createBufferSource()
      src.buffer = audioBuf
      src.connect(ctx.destination)
      const startAt = Math.max(nextStart, ctx.currentTime + 0.03)
      src.start(startAt)
      nextStart = startAt + audioBuf.duration
      scheduledSources.push(src)

      if (!played) { played = true; activeState = 'playing'; emit() }
    }
  }

  if (!played) {
    const e = new Error('No audio was produced.')
    throw e
  }

  // Stream done: go idle shortly after the last scheduled buffer finishes.
  const ctxNow = ctx.currentTime
  const tailMs = Math.max(0, (nextStart - ctxNow) * 1000) + 120
  idleTimer = setTimeout(() => { if (activeId === id) reset() }, tailMs)
}

// Blob playback (fallback): one WAV via a single Audio().
async function blobSpeak(id, markdown) {
  let r
  try {
    r = await fetch(`${API_BASE}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: markdown, voice: settings.voice,
        speed: settings.speed, read_code: settings.readCode,
      }),
    })
  } catch (err) {
    reset()
    throw new Error('Could not reach the TTS endpoint.')
  }
  if (!r.ok) {
    reset()
    const e = await r.json().catch(() => ({ detail: 'TTS failed' }))
    throw new Error(e.detail || `TTS failed (${r.status})`)
  }

  const blob = await r.blob()
  if (activeId !== id) return

  const url = URL.createObjectURL(blob)
  lastUrl = url
  const a = ensureAudio()
  a.src = url
  a.playbackRate = 1.0 // speed is applied server-side by Kokoro
  activeState = 'playing'
  emit()
  try {
    await a.play()
  } catch (err) {
    reset()
    throw new Error('Playback was blocked by the browser.')
  }
}

// --- STT capture -------------------------------------------------------------

function pickMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    '',
  ]
  if (typeof MediaRecorder === 'undefined') return ''
  for (const t of candidates) {
    if (t === '' || MediaRecorder.isTypeSupported(t)) return t
  }
  return ''
}

// getUserMedia requires a secure context (HTTPS or localhost). On a plain-HTTP
// LAN IP it's undefined — the same secure-context constraint the clipboard
// fallback already handles. We surface a clear message rather than throwing.
export function micSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia &&
            typeof MediaRecorder !== 'undefined')
}

// Returns a recorder handle: { stop() } that resolves the onResult/onError
// callbacks when transcription completes. Refine defaults to 'regex'.
export async function startDictation({ onResult, onError, refine = 'regex' }) {
  if (!micSupported()) {
    onError?.('Microphone needs a secure context (localhost or HTTPS).')
    return null
  }
  let stream
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  } catch (err) {
    onError?.(`Microphone unavailable: ${err.message || err.name}`)
    return null
  }

  const mimeType = pickMimeType()
  const mr = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
  const chunks = []

  mr.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data) }
  mr.onstop = async () => {
    stream.getTracks().forEach(t => t.stop())
    const blob = new Blob(chunks, { type: mr.mimeType || 'audio/webm' })
    if (!blob.size) { onError?.('No audio captured.'); return }
    try {
      const fd = new FormData()
      const ext = (mr.mimeType || '').includes('ogg') ? 'ogg' : 'webm'
      fd.append('file', blob, `clip.${ext}`)
      fd.append('refine', refine)
      const r = await fetch(`${API_BASE}/stt`, { method: 'POST', body: fd })
      if (!r.ok) {
        const e = await r.json().catch(() => ({ detail: 'Transcription failed' }))
        throw new Error(e.detail || `STT failed (${r.status})`)
      }
      const d = await r.json()
      if (d.timings) {
        // Read these in devtools to see where STT time goes. The first dictation
        // after a cold start carries model+VAD load in transcribe_ms.
        console.debug('[nchat stt]', d.timings)
      }
      onResult?.(d.text || d.raw || '')
    } catch (err) {
      onError?.(err.message || 'Transcription failed.')
    }
  }

  mr.start()
  return {
    stop: () => { try { mr.stop() } catch {} },
  }
}