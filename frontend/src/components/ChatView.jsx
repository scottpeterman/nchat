import React, { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'
import { startDictation, micSupported } from '../voice'

function formatTokens(n) {
  if (!n) return ''
  return n >= 1000 ? `~${(n / 1000).toFixed(1)}k tok` : `~${n} tok`
}

// Hold-to-talk dictation. Captures a clip, sends it to /api/stt, and hands the
// transcribed text back to the parent to append into the textarea. It never
// submits — review before send. Reflects recording/transcribing state, and
// fails soft (a clear message) when the mic isn't available.
function MicButton({ onText, onError, disabled }) {
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const recorderRef = useRef(null)
  const supported = micSupported()

  const start = async () => {
    onError('')
    const handle = await startDictation({
      onResult: (text) => { setBusy(false); if (text) onText(text) },
      onError: (msg) => { setBusy(false); setRecording(false); onError(msg) },
    })
    if (handle) {
      recorderRef.current = handle
      setRecording(true)
    }
  }

  const stop = () => {
    if (recorderRef.current) {
      setBusy(true)
      recorderRef.current.stop()
      recorderRef.current = null
    }
    setRecording(false)
  }

  const handleClick = () => {
    if (disabled && !recording) return
    recording ? stop() : start()
  }

  const title = !supported
    ? 'Mic needs a secure context (localhost or HTTPS)'
    : recording ? 'Stop and transcribe'
    : busy ? 'Transcribing…'
    : 'Dictate (speech to text)'

  return (
    <button
      type="button"
      className={`mic-btn ${recording ? 'recording' : ''} ${busy ? 'busy' : ''}`}
      onClick={handleClick}
      disabled={(disabled && !recording) || busy || !supported}
      title={title}
    >
      {busy ? '···' : recording ? '◼' : '🎙'}
    </button>
  )
}

export default function ChatView({ messages, onSend, onUpload, onStop, activeConvId, isStreaming, model }) {
  const [input, setInput] = useState('')
  const [pendingFiles, setPendingFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [voiceError, setVoiceError] = useState('')
  const messagesEndRef = useRef(null)
  const containerRef = useRef(null)
  const autoScrollRef = useRef(true)
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  // Track whether the user is parked at the bottom. If they've scrolled up to
  // read, we stop yanking them down on each update.
  const handleScroll = () => {
    const el = containerRef.current
    if (!el) return
    autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  useEffect(() => {
    const el = containerRef.current
    if (el && autoScrollRef.current) {
      el.scrollTop = el.scrollHeight   // instant — no animation to fight the stream
    }
  }, [messages])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px'
    }
  }, [input])

  const handleSubmit = (e) => {
    e.preventDefault()
    if ((!input.trim() && pendingFiles.length === 0) || isStreaming || uploading) return
    onSend(input.trim(), pendingFiles)
    setInput('')
    setPendingFiles([])
    setUploadError('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleFiles = async (fileList) => {
    const files = Array.from(fileList || [])
    if (files.length === 0) return
    setUploadError('')
    setUploading(true)
    for (const file of files) {
      try {
        const rec = await onUpload(file, activeConvId)
        setPendingFiles(prev => [...prev, rec])
      } catch (err) {
        setUploadError(err.message || `Could not upload ${file.name}`)
      }
    }
    setUploading(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const removeFile = (id) => {
    setPendingFiles(prev => prev.filter(f => f.id !== id))
  }

  // Append dictated text to whatever's already in the box, with a separating
  // space. Does not submit — the user reviews and edits before sending.
  const appendDictation = (text) => {
    setVoiceError('')
    setInput(prev => (prev && !prev.endsWith(' ') ? prev + ' ' : prev) + text)
    textareaRef.current?.focus()
  }

  const disabled = isStreaming || uploading

  return (
    <div className="chat-view">
      <div className="messages-container" ref={containerRef} onScroll={handleScroll}>
        {messages.length === 0 ? (
          <div className="empty-chat">
            <div className="empty-chat-icon">⌘</div>
            <h2>nChat</h2>
            <p>Local LLM chat powered by Ollama</p>
            <p className="model-hint">Model: <strong>{model || 'none selected'}</strong></p>
          </div>
        ) : (
          messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="input-area" onSubmit={handleSubmit}>
        {uploadError && (
          <div className="upload-error">{uploadError}</div>
        )}
        {voiceError && (
          <div className="upload-error">{voiceError}</div>
        )}
        {pendingFiles.length > 0 && (
          <div className="pending-files">
            {pendingFiles.map(f => (
              <div className="file-chip" key={f.id} title={`${f.filename} · ${formatTokens(f.est_tokens)}`}>
                <span className="file-chip-icon">📄</span>
                <span className="file-chip-name">{f.filename}</span>
                <span className="file-chip-tokens">{formatTokens(f.est_tokens)}</span>
                <button type="button" className="file-chip-remove" onClick={() => removeFile(f.id)} title="Remove">×</button>
              </div>
            ))}
          </div>
        )}
        <div className="input-wrapper">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => handleFiles(e.target.files)}
          />
          <button
            type="button"
            className="attach-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            title="Attach clear-text file"
          >
            {uploading ? '…' : '+'}
          </button>
          <MicButton
            onText={appendDictation}
            onError={setVoiceError}
            disabled={disabled}
          />
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isStreaming ? 'Waiting for response...' : 'Type a message... (Enter to send, Shift+Enter for newline)'}
            disabled={isStreaming}
            rows={1}
          />
          {isStreaming ? (
            <button
              type="button"
              className="send-btn stop"
              onClick={onStop}
              title="Stop generating"
            >
              ■
            </button>
          ) : (
            <button
              type="submit"
              className="send-btn"
              disabled={(!input.trim() && pendingFiles.length === 0) || disabled}
            >
              →
            </button>
          )}
        </div>
      </form>
    </div>
  )
}