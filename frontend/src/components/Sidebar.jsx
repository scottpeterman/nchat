import React from 'react'

function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  const now = new Date()
  const diff = now - d
  if (diff < 86400000) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  if (diff < 604800000) {
    return d.toLocaleDateString([], { weekday: 'short' })
  }
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export default function Sidebar({ conversations, activeConvId, onSelect, onNew, onDelete, isOpen, onToggle }) {
  const handleSelect = (convId) => {
    onSelect(convId)
    // Close sidebar on mobile after selection
    if (window.innerWidth <= 768) {
      onToggle()
    }
  }

  return (
    <>
      <div
        className={`sidebar-overlay ${isOpen ? 'visible' : ''}`}
        onClick={onToggle}
      />
      <aside className={`sidebar ${isOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-header">
          <span className="sidebar-brand">nChat</span>
          <button className="toggle-sidebar" onClick={onToggle} title="Close sidebar">✕</button>
        </div>

        <button className="new-chat-btn" onClick={onNew}>
          <span>+</span> New Chat
        </button>

        <div className="conversation-list">
          {conversations.length === 0 ? (
            <div className="empty-conversations">No conversations yet</div>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.id}
                className={`conversation-item ${conv.id === activeConvId ? 'active' : ''}`}
                onClick={() => handleSelect(conv.id)}
              >
                <div className="conv-title">{conv.title}</div>
                <div className="conv-meta">
                  <span className="conv-date">{formatDate(conv.updated_at)}</span>
                  <button
                    className="conv-delete"
                    onClick={(e) => { e.stopPropagation(); onDelete(conv.id) }}
                    title="Delete conversation"
                  >
                    ×
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-footer-text">Local LLM Interface</div>
        </div>
      </aside>
    </>
  )
}