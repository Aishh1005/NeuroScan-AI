import React from 'react'
import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 dark:bg-slate-950 text-center px-4">
      <div className="text-5xl mb-4">🧠</div>
      <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">Page not found</h1>
      <p className="text-slate-500 dark:text-slate-400 mb-6">The page you're looking for doesn't exist.</p>
      <Link to="/" className="btn-primary">Go home</Link>
    </div>
  )
}
