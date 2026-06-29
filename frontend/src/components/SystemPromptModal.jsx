import React, { useState } from 'react'

export default function SystemPromptModal({
  value, presets, onApply, onSavePreset, onDeletePreset, onClose,
}) {
  const [draft, setDraft] = useState(value || '')
  const [presetName, setPresetName] = useState('')
  const [saving, setSaving] = useState(false)

  const handleSavePreset = async () => {
    const name = presetName.trim()
    if (!name || !draft.trim()) return
    setSaving(true)
    try {
      await onSavePreset(name, draft)
      setPresetName('')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>System Prompt</h3>
          <button className="modal-close" onClick={onClose} title="Close">×</button>
        </div>

        <div className="modal-body">
          <p className="modal-hint">
            Sets the persona for this conversation. Sent to the model ahead of the chat history.
          </p>

          {presets.length > 0 && (
            <div className="preset-list">
              {presets.map(p => (
                <div className="preset-row" key={p.id}>
                  <button
                    className="preset-use"
                    onClick={() => setDraft(p.content)}
                    title={p.content}
                  >
                    {p.name}
                  </button>
                  <button
                    className="preset-delete"
                    onClick={() => onDeletePreset(p.id)}
                    title="Delete preset"
                  >×</button>
                </div>
              ))}
            </div>
          )}

          <textarea
            className="system-textarea"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="e.g. You are a senior network engineer. Be protocol-accurate and call out config errors explicitly."
            rows={8}
          />

          <div className="save-preset-row">
            <input
              type="text"
              className="preset-name-input"
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              placeholder="Save current as preset (name)…"
              onKeyDown={(e) => { if (e.key === 'Enter') handleSavePreset() }}
            />
            <button
              className="btn-secondary"
              onClick={handleSavePreset}
              disabled={!presetName.trim() || !draft.trim() || saving}
            >
              Save preset
            </button>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-ghost" onClick={() => setDraft('')}>Clear</button>
          <div className="modal-footer-right">
            <button className="btn-ghost" onClick={onClose}>Cancel</button>
            <button
              className="btn-primary"
              onClick={() => { onApply(draft); onClose() }}
            >
              Apply
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}