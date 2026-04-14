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
  const [result, setResult]     = useState<PredictionResult | null>(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const [preview, setPreview]   = useState<string | null>(null)

  const handleUpload = async (file: File) => {
    setLoading(true)
    setError(null)
    setResult(null)

    // Show image preview
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
    <main className="min-h-screen bg-gray-50">
      <Header />
      <div className="max-w-4xl mx-auto px-4 py-10">
        <UploadSection
          onUpload={handleUpload}
          loading={loading}
          preview={preview}
          onReset={handleReset}
        />
        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
            {error}
          </div>
        )}
        {result && (
          <ResultCard result={result} preview={preview} />
        )}
      </div>
    </main>
  )
}