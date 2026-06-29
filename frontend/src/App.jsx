import React, { useState, useEffect, useCallback, useRef } from 'react'
import Sidebar from './components/Sidebar'
import ChatView from './components/ChatView'
import ModelSelector from './components/ModelSelector'
import SystemPromptModal from './components/SystemPromptModal'
import { fetchVoices, setVoiceSettings } from './voice'

const API_BASE = '/api'

function uuid() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
}

export default function App() {
  const [conversations, setConversations] = useState([])
  const [activeConvId, setActiveConvId] = useState(null)
  const [messages, setMessages] = useState([])
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [ollamaStatus, setOllamaStatus] = useState('checking')
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 768)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [presets, setPresets] = useState([])
  const [showSystemModal, setShowSystemModal] = useState(false)
  const [thinkEnabled, setThinkEnabled] = useState(true)
  const [searchEnabled, setSearchEnabled] = useState(false)
  const [voices, setVoices] = useState([])
  const [selectedVoice, setSelectedVoice] = useState('af_heart')
  const abortRef = useRef(null)

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  // Check health & load models
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(data => {
        setOllamaStatus(data.ollama ? 'connected' : 'disconnected')
      })
      .catch(() => setOllamaStatus('error'))

    fetch(`${API_BASE}/models`)
      .then(r => r.json())
      .then(data => {
        setModels(data.models || [])
        if (data.models?.length > 0 && !selectedModel) {
          // Prefer qwen2.5-coder if available
          const qwen = data.models.find(m => m.name.includes('qwen2.5-coder'))
          setSelectedModel(qwen ? qwen.name : data.models[0].name)
        }
      })
      .catch(() => {})

    loadConversations()
    loadPresets()

    // Populate the voice catalog (served even without the engine loaded).
    fetchVoices().then(vs => {
      setVoices(vs)
      if (vs.length && !vs.find(v => v.id === selectedVoice)) {
        setSelectedVoice(vs[0].id)
        setVoiceSettings({ voice: vs[0].id })
      }
    })
  }, [])

  // Keep the shared voice controller in sync with the dropdown selection.
  const onVoiceChange = useCallback((voiceId) => {
    setSelectedVoice(voiceId)
    setVoiceSettings({ voice: voiceId })
  }, [])

  const loadPresets = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/prompts`)
      const data = await r.json()
      setPresets(data.prompts || [])
    } catch {}
  }, [])

  const loadConversations = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/conversations?limit=50`)
      const data = await r.json()
      setConversations(data.conversations || [])
    } catch {}
  }, [])

  const loadMessages = useCallback(async (convId) => {
    try {
      const r = await fetch(`${API_BASE}/conversations/${convId}/messages`)
      const data = await r.json()
      setMessages(data.messages || [])
    } catch {}
  }, [])

  const selectConversation = useCallback(async (convId) => {
    setActiveConvId(convId)
    await loadMessages(convId)
    const conv = conversations.find(c => c.id === convId)
    if (conv?.model) {
      setSelectedModel(conv.model)
    }
    setSystemPrompt(conv?.system_prompt || '')
  }, [conversations, loadMessages])

  const newConversation = useCallback(() => {
    setActiveConvId(null)
    setMessages([])
  }, [])

  const deleteConversation = useCallback(async (convId) => {
    try {
      await fetch(`${API_BASE}/conversations/${convId}`, { method: 'DELETE' })
      if (activeConvId === convId) {
        setActiveConvId(null)
        setMessages([])
      }
      await loadConversations()
    } catch {}
  }, [activeConvId, loadConversations])

  const uploadFile = useCallback(async (file, convId) => {
    const form = new FormData()
    form.append('file', file)
    if (convId) form.append('conversation_id', convId)
    const r = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: 'Upload failed' }))
      throw new Error(err.detail || `Upload failed (${r.status})`)
    }
    return r.json()  // { id, filename, size_bytes, est_tokens, created_at }
  }, [])

  // Commit a system prompt; persist to the conversation if one already exists.
  // For a brand-new chat, it's held in state and saved server-side at send time.
  const applySystemPrompt = useCallback(async (text) => {
    setSystemPrompt(text)
    if (activeConvId) {
      try {
        await fetch(`${API_BASE}/conversations/${activeConvId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ system_prompt: text }),
        })
      } catch {}
    }
  }, [activeConvId])

  const savePreset = useCallback(async (name, content) => {
    await fetch(`${API_BASE}/prompts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, content }),
    })
    await loadPresets()
  }, [loadPresets])

  const deletePreset = useCallback(async (id) => {
    try {
      await fetch(`${API_BASE}/prompts/${id}`, { method: 'DELETE' })
      await loadPresets()
    } catch {}
  }, [loadPresets])

  const sendMessage = useCallback(async (content, files = []) => {
    if ((!content.trim() && files.length === 0) || isStreaming) return

    const fileIds = files.map(f => f.id)

    // Optimistic UI: add user message immediately (with file chips)
    const userMsg = {
      id: uuid(), role: 'user', content,
      files: files.map(f => ({ id: f.id, filename: f.filename, est_tokens: f.est_tokens })),
      created_at: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])
    setIsStreaming(true)

    // Placeholder for assistant response
    const assistantId = uuid()
    setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '', streaming: true, created_at: new Date().toISOString() }])

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          conversation_id: activeConvId,
          message: content,
          model: selectedModel,
          system_prompt: systemPrompt,
          think: thinkEnabled,
          web_search: searchEnabled,
          file_ids: fileIds,
        }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullText = ''
      let fullThinking = ''
      let stats = null
      let meta = null
      let searchSources = null
      let newConvId = activeConvId

      // Batch token rendering: updating React state on every token forces a full
      // re-render (and re-highlight of the whole code block) per token, which is
      // what made the stream jerky. Flush at most every ~60ms instead.
      let lastFlush = 0
      const FLUSH_MS = 60
      const flush = () => {
        setMessages(prev => prev.map(m =>
          m.id === assistantId ? { ...m, content: fullText, thinking: fullThinking } : m
        ))
      }
      const maybeFlush = () => {
        const now = performance.now()
        if (now - lastFlush > FLUSH_MS) {
          lastFlush = now
          flush()
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))

            if (data.type === 'meta') {
              meta = { num_ctx: data.num_ctx, prompt_tokens_est: data.prompt_tokens_est }
              if (data.conversation_id) {
                newConvId = data.conversation_id
                if (!activeConvId) setActiveConvId(newConvId)
              }
            } else if (data.type === 'thinking') {
              fullThinking += data.thinking
              maybeFlush()
            } else if (data.type === 'search') {
              searchSources = data.sources || []
              setMessages(prev => prev.map(m =>
                m.id === assistantId
                  ? { ...m, sources: searchSources, searchNotice: data.notice || '' }
                  : m
              ))
            } else if (data.type === 'token') {
              fullText += data.token
              maybeFlush()
            } else if (data.type === 'done') {
              stats = data
            } else if (data.type === 'error') {
              fullText = `**Error:** ${data.error}`
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? { ...m, content: fullText, streaming: false } : m
              ))
            }
          } catch {}
        }
      }

      // Make sure the final tokens since the last throttled flush are shown
      flush()

      // Finalize — fold num_ctx into the stats line
      const finalStats = (stats || meta) ? { ...(stats || {}), ...(meta || {}) } : null
      const truncated = stats?.done_reason === 'length'
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, streaming: false, stats: finalStats, truncated,
              sources: searchSources ?? m.sources }
          : m
      ))

      // Self-heal: Ollama reported it generated tokens, but none rendered into
      // the bubble live (stats present, content empty). Reconcile from the DB,
      // which is the source of truth, so the answer isn't lost from the view.
      if (!fullText && stats && stats.tokens_eval > 0) {
        const convToLoad = newConvId || activeConvId
        if (convToLoad) await loadMessages(convToLoad)
      }

      await loadConversations()

    } catch (err) {
      if (err.name === 'AbortError') {
        // User stopped generation — keep whatever streamed in, mark it stopped.
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? { ...m, streaming: false, stopped: true }
            : m
        ))
        await loadConversations()
      } else {
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? { ...m, content: `**Connection error:** ${err.message}`, streaming: false }
            : m
        ))
      }
    } finally {
      abortRef.current = null
      setIsStreaming(false)
    }
  }, [activeConvId, selectedModel, systemPrompt, thinkEnabled, searchEnabled, isStreaming, loadConversations, loadMessages])

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeConvId={activeConvId}
        onSelect={selectConversation}
        onNew={newConversation}
        onDelete={deleteConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      <main className={`main-content ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
        <header className="top-bar">
          <button className="toggle-sidebar hamburger" onClick={() => setSidebarOpen(!sidebarOpen)} title="Toggle sidebar">
            ☰
          </button>
          <div className="top-bar-left">
            <h1 className="app-title">nChat</h1>
            <span className={`status-dot ${ollamaStatus}`} title={`Ollama: ${ollamaStatus}`} />
          </div>
          <div className="top-bar-right">
            <button
              className={`system-prompt-btn ${thinkEnabled ? 'active' : ''}`}
              onClick={() => setThinkEnabled(v => !v)}
              title={thinkEnabled ? 'Reasoning shown (thinking models) — click to turn off' : 'Show model reasoning (thinking models)'}
            >
              <span className="system-prompt-dot" />
              {thinkEnabled ? 'Think on' : 'Think off'}
            </button>
            <button
              className={`system-prompt-btn ${searchEnabled ? 'active' : ''}`}
              onClick={() => setSearchEnabled(v => !v)}
              title={searchEnabled ? 'Web search on — answers grounded in live results' : 'Search the web before answering'}
            >
              <span className="system-prompt-dot" />
              {searchEnabled ? 'Search on' : 'Search off'}
            </button>
            <button
              className={`system-prompt-btn ${systemPrompt ? 'active' : ''}`}
              onClick={() => setShowSystemModal(true)}
              title={systemPrompt ? 'System prompt set — click to edit' : 'Set a system prompt / persona'}
            >
              <span className="system-prompt-dot" />
              {systemPrompt ? 'Persona set' : 'System'}
            </button>
            <ModelSelector
              models={models}
              selected={selectedModel}
              onChange={setSelectedModel}
            />
            {voices.length > 0 && (
              <select
                className="voice-selector"
                value={selectedVoice}
                onChange={(e) => onVoiceChange(e.target.value)}
                title="Voice for reading responses aloud"
              >
                {voices.map(v => (
                  <option key={v.id} value={v.id}>{v.label}</option>
                ))}
              </select>
            )}
          </div>
        </header>
        <ChatView
          messages={messages}
          onSend={sendMessage}
          onUpload={uploadFile}
          onStop={stopGeneration}
          activeConvId={activeConvId}
          isStreaming={isStreaming}
          model={selectedModel}
        />
      </main>
      {showSystemModal && (
        <SystemPromptModal
          value={systemPrompt}
          presets={presets}
          onApply={applySystemPrompt}
          onSavePreset={savePreset}
          onDeletePreset={deletePreset}
          onClose={() => setShowSystemModal(false)}
        />
      )}
    </div>
  )
}