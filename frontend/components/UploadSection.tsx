'use client'

import { useRef, useState } from 'react'

interface Props {
  onUpload : (file: File) => void
  loading  : boolean
  preview  : string | null
  onReset  : () => void
}

export default function UploadSection({ onUpload, loading, preview, onReset }: Props) {
  const inputRef            = useRef<HTMLInputElement>(null)
  const [dragging, setDrag] = useState(false)

  const handleFile = (file: File) => {
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
      alert('Only JPEG and PNG images are allowed')
      return
    }
    onUpload(file)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDrag(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div style={{
      background: '#0d1424',
      border: '1px solid #1e3a5f',
      borderRadius: 16,
      padding: 24
    }}>
      <h2 style={{ fontSize: 15, fontWeight: 600, color: '#f1f5f9', marginBottom: 16 }}>
        Upload MRI Scan
      </h2>

      {!preview ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          style={{
            border: `2px dashed ${dragging ? '#3b82f6' : '#1e3a5f'}`,
            borderRadius: 12,
            padding: '48px 24px',
            textAlign: 'center',
            cursor: 'pointer',
            background: dragging ? '#0f1f3d' : '#070d1a',
            transition: 'all 0.2s'
          }}
        >
          <div style={{ fontSize: 36, marginBottom: 12 }}>🧠</div>
          <p style={{ fontSize: 14, fontWeight: 500, color: '#cbd5e1' }}>
            Drop MRI image here or click to browse
          </p>
          <p style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>
            Supports JPEG, PNG — max 10MB
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
            }}
          />
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ borderRadius: 12, overflow: 'hidden', border: '1px solid #1e3a5f' }}>
            <img src={preview} alt="MRI Preview"
              style={{ width: '100%', maxHeight: 280, objectFit: 'contain', background: '#000' }} />
          </div>
          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, padding: '12px 0' }}>
              <div style={{
                width: 18, height: 18,
                border: '2px solid #3b82f6',
                borderTopColor: 'transparent',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite'
              }} />
              <span style={{ fontSize: 13, color: '#94a3b8' }}>Analysing MRI scan...</span>
              <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
            </div>
          ) : (
            <button onClick={onReset} style={{
              width: '100%', padding: '10px 0', fontSize: 13,
              color: '#94a3b8', background: 'transparent',
              border: '1px solid #1e3a5f', borderRadius: 8, cursor: 'pointer'
            }}>
              Upload another scan
            </button>
          )}
        </div>
      )}
    </div>
  )
}
