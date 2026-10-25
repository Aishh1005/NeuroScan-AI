import React, { useEffect, useState } from 'react'
import api from '../services/api'
import { Spinner, ErrorAlert } from '../components/Common.jsx'

export default function ModelPerformance() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    (async () => {
      try {
        const res = await api.get('/api/model/performance')
        setData(res.data)
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to load model performance.')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  if (loading) return <Spinner />

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Model Performance</h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm">Evaluation metrics computed from the actual trained model on the held-out test set.</p>
      </div>

      <ErrorAlert message={error} />

      <div className="grid sm:grid-cols-2 gap-4">
        <StatusCard label="Classifier checkpoint" present={data?.classifier_model_present} />
        <StatusCard label="Segmentation checkpoint (optional)" present={data?.segmentation_model_present} />
      </div>

      {!data?.available ? (
        <div className="card p-8 text-center">
          <p className="text-slate-500 dark:text-slate-400">{data?.message || 'Model evaluation results are not yet available.'}</p>
          <p className="text-xs text-slate-400 mt-2">
            Run <code className="bg-slate-100 dark:bg-slate-800 px-1 rounded">ml/training/evaluate.py</code> after training to populate this page with real metrics.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {data.models.map((m) => (
            <div key={m.model_name} className="card p-6 space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="font-semibold text-lg text-slate-900 dark:text-white">{m.model_name}</h3>
                <span className="text-xs text-slate-400">{new Date(m.trained_at).toLocaleDateString()}</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 text-center">
                <Metric label="Accuracy" value={m.accuracy} />
                <Metric label="Precision" value={m.precision} />
                <Metric label="Recall" value={m.recall} />
                <Metric label="F1-score" value={m.f1_score} />
                <Metric label="ROC-AUC" value={m.roc_auc} />
              </div>
              <div className="text-xs text-slate-500 dark:text-slate-400 space-y-1">
                <div><span className="font-medium">Architecture:</span> {m.architecture}</div>
                <div><span className="font-medium">Classes:</span> {m.num_classes}</div>
                <div><span className="font-medium">Train set:</span> {m.train_dataset}</div>
                <div><span className="font-medium">Test set:</span> {m.test_dataset}</div>
              </div>
              {m.confusion_matrix && (
                <div>
                  <h4 className="text-sm font-medium mb-2 text-slate-700 dark:text-slate-200">Confusion Matrix</h4>
                  <ConfusionMatrix cm={m.confusion_matrix} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function StatusCard({ label, present }) {
  return (
    <div className="card p-4 flex items-center justify-between">
      <span className="text-sm text-slate-600 dark:text-slate-300">{label}</span>
      <span className={`badge ${present ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300' : 'bg-slate-100 text-slate-500 dark:bg-slate-800'}`}>
        {present ? 'Present' : 'Not found'}
      </span>
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div>
      <div className="text-lg font-bold text-slate-900 dark:text-white">
        {value != null ? `${(value * 100).toFixed(1)}%` : 'N/A'}
      </div>
      <div className="text-xs text-slate-400">{label}</div>
    </div>
  )
}

function ConfusionMatrix({ cm }) {
  const { labels, matrix } = cm
  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-collapse">
        <thead>
          <tr>
            <th className="p-2"></th>
            {labels.map((l) => <th key={l} className="p-2 text-slate-500">{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="p-2 font-medium text-slate-500">{labels[i]}</td>
              {row.map((v, j) => (
                <td key={j} className={`p-2 text-center border border-slate-100 dark:border-slate-800 ${i === j ? 'bg-brand-50 dark:bg-brand-900/30 font-semibold' : ''}`}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
