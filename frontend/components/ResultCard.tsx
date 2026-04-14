'use client'

import { PredictionResult } from '@/app/page'

interface Props {
  result  : PredictionResult
  preview : string | null
}

const SEVERITY_STYLES: Record<string, string> = {
  high   : 'bg-red-100 text-red-700 border-red-200',
  medium : 'bg-yellow-100 text-yellow-700 border-yellow-200',
  none   : 'bg-green-100 text-green-700 border-green-200'
}

const SEVERITY_LABEL: Record<string, string> = {
  high   : 'High Severity',
  medium : 'Medium Severity',
  none   : 'No Tumor Detected'
}

const CLASS_COLORS: Record<string, string> = {
  glioma     : 'bg-red-500',
  meningioma : 'bg-yellow-500',
  notumor    : 'bg-green-500',
  pituitary  : 'bg-blue-500'
}

export default function ResultCard({ result }: Props) {
  const severityStyle = SEVERITY_STYLES[result.severity] || SEVERITY_STYLES.none

  return (
    <div className="mt-6 bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-5">

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wide mb-1">Prediction Result</p>
          <h3 className="text-xl font-bold text-gray-900">{result.label}</h3>
          <p className="text-sm text-gray-500 mt-1">{result.description}</p>
        </div>
        <span className={`text-xs font-medium px-3 py-1 rounded-full border ${severityStyle}`}>
          {SEVERITY_LABEL[result.severity]}
        </span>
      </div>

      {/* Confidence */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-600 font-medium">Confidence</span>
          <span className="font-bold text-gray-900">{result.confidence}%</span>
        </div>
        <div className="w-full bg-gray-100 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full ${CLASS_COLORS[result.predicted_class] || 'bg-blue-500'}`}
            style={{ width: `${result.confidence}%` }}
          />
        </div>
      </div>

      {/* All class scores */}
      <div>
        <p className="text-xs text-gray-400 uppercase tracking-wide mb-3">All Class Scores</p>
        <div className="space-y-2">
          {Object.entries(result.all_scores)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, score]) => (
              <div key={cls}>
                <div className="flex justify-between text-xs mb-1">
                  <span className={`font-medium capitalize ${cls === result.predicted_class ? 'text-gray-900' : 'text-gray-500'}`}>
                    {cls === result.predicted_class ? `▶ ${cls}` : cls}
                  </span>
                  <span className={cls === result.predicted_class ? 'font-bold text-gray-900' : 'text-gray-400'}>
                    {score}%
                  </span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full ${CLASS_COLORS[cls] || 'bg-gray-400'}`}
                    style={{ width: `${score}%` }}
                  />
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Footer */}
      <div className="pt-3 border-t border-gray-100 flex items-center justify-between">
        <p className="text-xs text-gray-400">
          Model: VGG16 {result.model_version}
        </p>
        <p className="text-xs text-red-400 font-medium">
          For research purposes only — not a medical diagnosis
        </p>
      </div>
    </div>
  )
}