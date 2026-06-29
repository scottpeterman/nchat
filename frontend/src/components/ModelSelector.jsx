import React from 'react'

function formatSize(bytes) {
  if (!bytes) return ''
  const gb = bytes / (1024 * 1024 * 1024)
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / (1024 * 1024)).toFixed(0)} MB`
}

export default function ModelSelector({ models, selected, onChange }) {
  if (models.length === 0) {
    return (
      <div className="model-selector">
        <select disabled>
          <option>No models available</option>
        </select>
      </div>
    )
  }

  return (
    <div className="model-selector">
      <label>Model:</label>
      <select value={selected} onChange={(e) => onChange(e.target.value)}>
        {models.map(m => (
          <option key={m.name} value={m.name}>
            {m.name} {m.parameter_size ? `(${m.parameter_size})` : ''} {m.size ? `· ${formatSize(m.size)}` : ''}
          </option>
        ))}
      </select>
    </div>
  )
}
