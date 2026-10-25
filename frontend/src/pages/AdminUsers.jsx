import React, { useEffect, useState } from 'react'
import api from '../services/api'
import { Spinner, ErrorAlert } from '../components/Common.jsx'

export default function AdminUsers() {
  const [users, setUsers] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState(null)

  const load = async () => {
    try {
      const res = await api.get('/api/admin/users')
      setUsers(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load users.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const toggleActive = async (id) => {
    setBusyId(id)
    try {
      await api.patch(`/api/admin/users/${id}/toggle-active`)
      await load()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update user status.')
    } finally {
      setBusyId(null)
    }
  }

  if (loading) return <Spinner />

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Users</h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm">Manage registered accounts.</p>
      </div>

      <ErrorAlert message={error} />

      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              <th className="py-3 px-4">Name</th>
              <th className="py-3 px-4">Email</th>
              <th className="py-3 px-4">Role</th>
              <th className="py-3 px-4">Status</th>
              <th className="py-3 px-4">Joined</th>
              <th className="py-3 px-4"></th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-b border-slate-100 dark:border-slate-800">
                <td className="py-3 px-4">{u.full_name}</td>
                <td className="py-3 px-4">{u.email}</td>
                <td className="py-3 px-4 capitalize">{u.role}</td>
                <td className="py-3 px-4">
                  <span className={`badge ${u.is_active ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300' : 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300'}`}>
                    {u.is_active ? 'Active' : 'Deactivated'}
                  </span>
                </td>
                <td className="py-3 px-4">{new Date(u.created_at).toLocaleDateString()}</td>
                <td className="py-3 px-4">
                  {u.role !== 'admin' && (
                    <button
                      onClick={() => toggleActive(u.id)}
                      disabled={busyId === u.id}
                      className="text-xs font-medium text-brand-600"
                    >
                      {u.is_active ? 'Deactivate' : 'Activate'}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
