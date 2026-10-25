import React from 'react'

export function Disclaimer({ className = '' }) {
  return (
    <div className={`text-xs text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300 ${className}`}>
      AI-assisted research result. This system is not a medical diagnosis and should not be used as a
      substitute for professional medical evaluation.
    </div>
  )
}

export function LowConfidenceNotice() {
  return (
    <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 dark:bg-amber-950/40 dark:border-amber-900 dark:text-amber-300">
      Low-confidence prediction. Professional review is recommended.
    </div>
  )
}

export function ConfidenceBadge({ level, confidence }) {
  const styles = {
    high: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
    moderate: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
    low: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
  }
  return (
    <span className={`badge ${styles[level] || styles.low}`}>
      {level?.toUpperCase()} · {(confidence * 100).toFixed(1)}%
    </span>
  )
}

export function StatusBadge({ status }) {
  const styles = {
    completed: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300',
    pending: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
    failed: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
  }
  return <span className={`badge ${styles[status] || styles.pending}`}>{status}</span>
}

export function Spinner() {
  return (
    <div className="flex items-center justify-center py-10">
      <div className="w-8 h-8 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

export function ErrorAlert({ message }) {
  if (!message) return null
  return (
    <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2 dark:bg-red-950/40 dark:border-red-900 dark:text-red-300">
      {message}
    </div>
  )
}
