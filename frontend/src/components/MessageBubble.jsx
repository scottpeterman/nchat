import React, { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { toggleSpeak, subscribe } from '../voice'

// If ReactMarkdown / SyntaxHighlighter throws on some model output, fall back to
// rendering the raw text instead of blanking the whole message. Resets on every
// content change so the final (good) state still renders richly.
class MarkdownBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, lastRaw: props.raw }
  }
  static getDerivedStateFromError() {
    return { hasError: true }
  }
  static getDerivedStateFromProps(props, state) {
    if (props.raw !== state.lastRaw) {
      return { hasError: false, lastRaw: props.raw }
    }
    return null
  }
  render() {
    if (this.state.hasError) {
      return <pre className="markdown-fallback">{this.props.raw}</pre>
    }
    return this.props.children
  }
}

async function copyText(text) {
  // navigator.clipboard only exists in secure contexts (HTTPS or localhost).
  // On a plain-HTTP LAN IP it's undefined, so fall back to the legacy path.
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {}
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.top = '-1000px'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.focus()
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    const ok = await copyText(text)
    if (ok) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <button className="copy-btn" onClick={handleCopy}>
      {copied ? '✓ Copied' : 'Copy'}
    </button>
  )
}

// Reads an assistant message aloud via the shared audio controller. Subscribes
// so it reflects loading/playing only when *this* message is the active one;
// any other message reading flips this back to idle automatically.
function ReadButton({ messageId, markdown }) {
  const [state, setState] = useState('idle')   // 'idle' | 'loading' | 'playing'
  const [error, setError] = useState('')

  useEffect(() => {
    return subscribe((id, st) => {
      setState(id === messageId ? st : 'idle')
    })
  }, [messageId])

  const handleClick = async () => {
    setError('')
    try {
      await toggleSpeak(messageId, markdown)
    } catch (e) {
      setError(e.message || 'Could not read this message.')
      setTimeout(() => setError(''), 4000)
    }
  }

  const label =
    state === 'loading' ? '··· Loading' :
    state === 'playing' ? '◼ Stop' :
    '▶ Read'

  return (
    <>
      <button
        className={`msg-action-btn ${state !== 'idle' ? 'active' : ''}`}
        onClick={handleClick}
        title={state === 'playing' ? 'Stop reading' : 'Read aloud'}
      >
        {label}
      </button>
      {error && <span className="msg-action-error" title={error}>⚠ {error}</span>}
    </>
  )
}

// Copies the whole message's Markdown (the per-code-block Copy still exists too).
function MessageCopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async () => {
    const ok = await copyText(text)
    if (ok) { setCopied(true); setTimeout(() => setCopied(false), 2000) }
  }
  return (
    <button className="msg-action-btn" onClick={handleCopy} title="Copy message">
      {copied ? '✓ Copied' : 'Copy'}
    </button>
  )
}

function CodeBlock({ children, className, ...props }) {
  const match = /language-(\w+)/.exec(className || '')
  const language = match ? match[1] : ''
  const raw = Array.isArray(children) ? children.join('') : String(children ?? '')
  const code = raw.replace(/\n$/, '')

  if (!match) {
    return <code className="inline-code" {...props}>{children}</code>
  }

  return (
    <div className="code-block">
      <div className="code-header">
        <span className="code-lang">{language}</span>
        <CopyButton text={code} />
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: '0 0 6px 6px',
          fontSize: '13px',
          lineHeight: '1.5',
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}

function ThinkingBlock({ thinking, streaming, hasContent }) {
  // Auto-expanded while the model is actively reasoning (no answer yet);
  // auto-collapses once the answer starts. User can override either way.
  const [userToggled, setUserToggled] = useState(null)
  const autoExpanded = streaming && !hasContent
  const expanded = userToggled === null ? autoExpanded : userToggled
  const active = streaming && !hasContent

  return (
    <div className="thinking-block">
      <button
        className="thinking-header"
        onClick={() => setUserToggled(!expanded)}
      >
        <span className="thinking-caret">{expanded ? '▾' : '▸'}</span>
        <span className="thinking-title">Thinking{active ? '…' : ''}</span>
      </button>
      {expanded && <div className="thinking-body">{thinking}</div>}
    </div>
  )
}

function MessageBubble({ message }) {
  const { role, content, streaming, stats, files, stopped, truncated, thinking, sources, searchNotice } = message

  return (
    <div className={`message ${role}`}>
      <div className="message-avatar">
        {role === 'user' ? 'You' : 'AI'}
      </div>
      <div className="message-body">
        {files && files.length > 0 && (
          <div className="message-files">
            {files.map(f => (
              <div className="file-chip readonly" key={f.id} title={f.filename}>
                <span className="file-chip-icon">📄</span>
                <span className="file-chip-name">{f.filename}</span>
                {f.est_tokens ? (
                  <span className="file-chip-tokens">
                    {f.est_tokens >= 1000 ? `~${(f.est_tokens / 1000).toFixed(1)}k tok` : `~${f.est_tokens} tok`}
                  </span>
                ) : null}
              </div>
            ))}
          </div>
        )}
        {thinking && (
          <ThinkingBlock thinking={thinking} streaming={streaming} hasContent={!!content} />
        )}
        <div className="message-content">
          {streaming && !content && !thinking ? (
            <span className="thinking">
              <span className="thinking-dot" />
              <span className="thinking-dot" />
              <span className="thinking-dot" />
              <span className="thinking-label">thinking…</span>
            </span>
          ) : content ? (
            <MarkdownBoundary raw={content}>
              <ReactMarkdown
                components={{
                  code: CodeBlock,
                }}
              >
                {content}
              </ReactMarkdown>
            </MarkdownBoundary>
          ) : null}
          {streaming && content && <span className="cursor-blink">▊</span>}
        </div>
        {(() => {
          // sources may arrive as a JSON string (reloaded from DB) or array (live)
          let srcs = sources
          if (typeof srcs === 'string' && srcs) {
            try { srcs = JSON.parse(srcs) } catch { srcs = null }
          }
          if (!srcs || srcs.length === 0) return null
          return (
            <div className="message-sources">
              <div className="sources-label">Sources</div>
              <ol className="sources-list">
                {srcs.map((s, i) => (
                  <li key={i}>
                    <a href={s.url} target="_blank" rel="noopener noreferrer" title={s.url}>
                      {s.title || s.url}
                    </a>
                  </li>
                ))}
              </ol>
            </div>
          )
        })()}
        {searchNotice && !streaming && (
          <div className="search-notice">{searchNotice}</div>
        )}
        {stopped && !streaming && (
          <div className="stopped-note">Stopped</div>
        )}
        {role === 'assistant' && !streaming && content && (
          <div className="message-actions">
            <ReadButton messageId={String(message.id)} markdown={content} />
            <MessageCopyButton text={content} />
          </div>
        )}
        {truncated && !streaming && (
          <div className="truncated-note">
            ⚠ Hit the context limit — this response is cut off. Ask it to “continue”, or raise the context size.
          </div>
        )}
        {stats && !streaming && (
          <div className="message-stats">
            {stats.tokens_per_sec > 0 && (
              <span>{stats.tokens_per_sec} tok/s</span>
            )}
            {stats.tokens_eval > 0 && (
              <span>{stats.tokens_eval} tokens</span>
            )}
            {stats.duration_ms > 0 && (
              <span>{(stats.duration_ms / 1000).toFixed(1)}s</span>
            )}
            {stats.num_ctx > 0 && (
              <span title="Context window sized for this request">ctx {stats.num_ctx >= 1024 ? `${(stats.num_ctx / 1024).toFixed(0)}k` : stats.num_ctx}</span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// Memoized: typing in the input re-renders ChatView, but message bubbles (and
// their syntax highlighting) should only re-render when their own message changes.
export default React.memo(MessageBubble)