'use client'

import { PredictionResult } from '@/app/page'

interface Props {
  result  : PredictionResult
  preview : string | null
}

const SEVERITY_STYLES: Record<string, { bg: string; color: string; label: string }> = {
  high   : { bg: '#1a0a0a', color: '#f87171', label: 'High Severity' },
  medium : { bg: '#1a1500', color: '#fbbf24', label: 'Medium Severity' },
  none   : { bg: '#0a1a0a', color: '#4ade80', label: 'No Tumor Detected' }
}

const CLASS_COLORS: Record<string, string> = {
  glioma     : '#ef4444',
  meningioma : '#f59e0b',
  notumor    : '#22c55e',
  pituitary  : '#3b82f6'
}

export default function ResultCard({ result }: Props) {
  const sev = SEVERITY_STYLES[result.severity] || SEVERITY_STYLES.none

  return (
    <div style={{
      marginTop: 20,
      background: '#0d1424',
      border: '1px solid #1e3a5f',
      borderRadius: 16,
      padding: 24,
      display: 'flex',
      flexDirection: 'column',
      gap: 20
    }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <p style={{ fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            Prediction Result
          </p>
          <h3 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9' }}>{result.label}</h3>
          <p style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{result.description}</p>
        </div>
        <span style={{
          fontSize: 11, fontWeight: 600, padding: '4px 12px',
          borderRadius: 20, background: sev.bg, color: sev.color,
          border: `1px solid ${sev.color}33`, whiteSpace: 'nowrap'
        }}>
          {sev.label}
        </span>
      </div>

      {/* Confidence */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: 13, color: '#94a3b8' }}>Confidence</span>
          <span style={{ fontSize: 14, fontWeight: 700, color: '#f1f5f9' }}>{result.confidence}%</span>
        </div>
        <div style={{ background: '#1e293b', borderRadius: 99, height: 8 }}>
          <div style={{
            height: 8, borderRadius: 99,
            background: CLASS_COLORS[result.predicted_class] || '#3b82f6',
            width: `${result.confidence}%`,
            transition: 'width 0.6s ease'
          }} />
        </div>
      </div>

      {/* All scores */}
      <div>
        <p style={{ fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
          All Class Scores
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {Object.entries(result.all_scores)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, score]) => (
              <div key={cls}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{
                    fontSize: 12, fontWeight: cls === result.predicted_class ? 600 : 400,
                    color: cls === result.predicted_class ? '#f1f5f9' : '#64748b'
                  }}>
                    {cls === result.predicted_class ? `▶ ${cls}` : cls}
                  </span>
                  <span style={{
                    fontSize: 12,
                    color: cls === result.predicted_class ? '#f1f5f9' : '#475569',
                    fontWeight: cls === result.predicted_class ? 700 : 400
                  }}>
                    {score}%
                  </span>
                </div>
                <div style={{ background: '#1e293b', borderRadius: 99, height: 4 }}>
                  <div style={{
                    height: 4, borderRadius: 99,
                    background: CLASS_COLORS[cls] || '#475569',
                    width: `${score}%`,
                    opacity: cls === result.predicted_class ? 1 : 0.4
                  }} />
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Footer */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        paddingTop: 16, borderTop: '1px solid #1e293b'
      }}>
        <span style={{ fontSize: 11, color: '#334155' }}>Model: VGG16 {result.model_version}</span>
        <span style={{ fontSize: 11, color: '#ef4444' }}>For research purposes only — not a medical diagnosis</span>
      </div>
    </div>
  )
}
