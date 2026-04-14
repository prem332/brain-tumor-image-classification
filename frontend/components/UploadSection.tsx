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
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
      <h2 className="text-base font-semibold text-gray-800 mb-4">
        Upload MRI Scan
      </h2>

      {!preview ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors
            ${dragging
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
            }`}
        >
          <div className="text-4xl mb-3">🧠</div>
          <p className="text-sm font-medium text-gray-700">
            Drop MRI image here or click to browse
          </p>
          <p className="text-xs text-gray-400 mt-1">
            Supports JPEG, PNG — max 10MB
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
            }}
          />
        </div>
      ) : (
        <div className="space-y-4">
          <div className="rounded-xl overflow-hidden border border-gray-200">
            <img
              src={preview}
              alt="MRI Preview"
              className="w-full max-h-72 object-contain bg-black"
            />
          </div>
          {loading ? (
            <div className="flex items-center justify-center gap-3 py-3">
              <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-gray-600">Analyzing MRI scan...</span>
            </div>
          ) : (
            <button
              onClick={onReset}
              className="w-full py-2 text-sm text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Upload another scan
            </button>
          )}
        </div>
      )}
    </div>
  )
}
