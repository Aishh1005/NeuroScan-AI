import React, { useEffect, useState } from 'react'
import api from '../services/api'
import { Spinner, ErrorAlert } from '../components/Common.jsx'

export default function AdminLogs() {
  const [logs, setLogs] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    (async () => {
      try {
        const res = await api.get('/api/admin/logs')
        setLogs(res.data)
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to load system logs.')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  if (loading) return <Spinner />

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">System Logs</h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm">Recent authentication, analysis, and admin actions.</p>
      </div>

      <ErrorAlert message={error} />

      <div className="card divide-y divide-slate-100 dark:divide-slate-800">
        {logs.map((l) => (
          <div key={l.id} className="p-4 flex justify-between items-start gap-4">
            <div>
              <div className="font-medium text-sm text-slate-800 dark:text-slate-100">{l.action}</div>
              {l.detail && <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{l.detail}</div>}
              {l.user_id && <div className="text-xs text-slate-400 mt-0.5 font-mono">user: {l.user_id.slice(0, 8)}…</div>}
            </div>
            <span className="text-xs text-slate-400 whitespace-nowrap">{new Date(l.created_at).toLocaleString()}</span>
          </div>
        ))}
        {logs.length === 0 && <p className="text-sm text-slate-400 text-center py-10">No log entries yet.</p>}
      </div>
    </div>
  )
}
