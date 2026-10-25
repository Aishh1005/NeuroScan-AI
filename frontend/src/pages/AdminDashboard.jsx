import React, { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import api from '../services/api'
import { Spinner, ErrorAlert, StatusBadge } from '../components/Common.jsx'

export default function AdminDashboard() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    (async () => {
      try {
        const res = await api.get('/api/admin/statistics')
        setStats(res.data)
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to load admin statistics.')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  if (loading) return <Spinner />

  const classData = stats ? Object.entries(stats.class_distribution).map(([name, count]) => ({ name, count })) : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Admin Dashboard</h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm">System-wide activity overview.</p>
      </div>

      <ErrorAlert message={error} />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Registered Users" value={stats?.total_users ?? 0} icon="👥" />
        <StatCard label="Total Analyses" value={stats?.total_analyses ?? 0} icon="📊" />
        <StatCard label="Tumor Positive" value={stats?.tumor_positive ?? 0} icon="⚠️" />
        <StatCard label="Tumor Negative" value={stats?.tumor_negative ?? 0} icon="✅" />
      </div>

      <div className="card p-5">
        <h3 className="font-semibold mb-3 text-slate-900 dark:text-white">Classification Distribution (System-wide)</h3>
        {classData.length ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={classData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-400 text-center py-10">No completed analyses yet.</p>
        )}
      </div>

      <div className="card p-5">
        <h3 className="font-semibold mb-3 text-slate-900 dark:text-white">Recent Analyses</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <th className="py-2 pr-4">Date</th>
                <th className="py-2 pr-4">User ID</th>
                <th className="py-2 pr-4">Prediction</th>
                <th className="py-2 pr-4">Confidence</th>
                <th className="py-2 pr-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {stats?.recent_analyses?.map((a) => (
                <tr key={a.id} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="py-2 pr-4">{new Date(a.created_at).toLocaleString()}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{a.user_id.slice(0, 8)}…</td>
                  <td className="py-2 pr-4">{a.predicted_class || '—'}</td>
                  <td className="py-2 pr-4">{a.confidence != null ? `${(a.confidence * 100).toFixed(1)}%` : '—'}</td>
                  <td className="py-2 pr-4"><StatusBadge status={a.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          {(!stats?.recent_analyses || stats.recent_analyses.length === 0) && (
            <p className="text-sm text-slate-400 text-center py-6">No analyses recorded yet.</p>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, icon }) {
  return (
    <div className="card p-4">
      <div className="text-xl mb-1">{icon}</div>
      <div className="text-2xl font-bold text-slate-900 dark:text-white">{value}</div>
      <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
    </div>
  )
}
