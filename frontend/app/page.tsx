'use client'

import { useState } from 'react'
import UploadSection from '@/components/UploadSection'
import ResultCard from '@/components/ResultCard'
import Header from '@/components/Header'

export interface PredictionResult {
  predicted_class: string
  label: string
  confidence: number
  severity: string
  description: string
  all_scores: Record<string, number>
  model_version: string
}

export default function Home() {
  const [result, setResult]   = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [preview, setPreview] = useState<string | null>(null)

  const handleUpload = async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)

    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(file)

    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/predict`,
        { method: 'POST', body: formData }
      )
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Prediction failed')
      }
      const data: PredictionResult = await response.json()
      setResult(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
    setPreview(null)
  }

  return (
    <main style={{ background: '#0a0f1e', minHeight: '100vh' }}>
      <Header />
      <div style={{
        maxWidth: 680,
        margin: '0 auto',
        padding: '40px 20px'
      }}>
        <UploadSection
          onUpload={handleUpload}
          loading={loading}
          preview={preview}
          onReset={handleReset}
        />
        {error && (
          <div style={{
            marginTop: 16, padding: 16, borderRadius: 12,
            background: '#1a0a0a', border: '1px solid #7f1d1d', color: '#fca5a5',
            fontSize: 13
          }}>
            {error}
          </div>
        )}
        {result && <ResultCard result={result} preview={preview} />}
      </div>
    </main>
  )
}
