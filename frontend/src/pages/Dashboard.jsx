import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api from '../services/api'
import { useAuth } from '../hooks/useAuth.jsx'
import {
  Spinner,
  ErrorAlert,
  StatusBadge,
  ConfidenceBadge,
} from '../components/Common.jsx'

export default function Dashboard() {
  const { user } = useAuth()

  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboard()
  }, [])

  async function loadDashboard() {
    try {
      setLoading(true)
      setError('')

      const [statsRes, historyRes] = await Promise.all([
        api.get('/api/analysis/dashboard'),
        api.get('/api/analysis/history'),
      ])

      setStats(statsRes.data || {})
      setHistory((historyRes.data || []).slice(0, 8))
    } catch (err) {
      console.error('Dashboard error:', err)

      setError(
        err.response?.data?.detail ||
        'Failed to load dashboard.'
      )
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-[#020b1c] flex items-center justify-center">
        <Spinner />
      </div>
    )
  }

  const firstName =
    user?.full_name?.trim()?.split(' ')[0] ||
    user?.name?.trim()?.split(' ')[0] ||
    'User'

  const totalAnalyses = stats?.total_analyses ?? 0

  const tumorDetected =
    stats?.tumor_detected_count ??
    stats?.tumor_count ??
    0

  const reports =
    stats?.reports_count ??
    history.length ??
    0

  const latest = history.length > 0 ? history[0] : null

  return (
    <div className="min-h-screen w-full bg-[#020b1c] text-white overflow-x-hidden">

      {/* =====================================================
          MAIN CONTENT
      ====================================================== */}

      <main className="w-full p-4 sm:p-6 lg:p-8">

        {/* =====================================================
            HEADER
        ====================================================== */}

        <div className="
          w-full
          flex
          flex-col
          lg:flex-row
          lg:items-center
          lg:justify-between
          gap-5
          mb-8
        ">

          <div>

            <h1 className="
              text-3xl
              lg:text-4xl
              font-bold
              text-white
            ">
              Welcome, {firstName} 👋
            </h1>

            <p className="
              text-slate-400
              mt-2
              text-sm
              sm:text-base
            ">
              Manage your MRI analyses, reports and results in one place.
            </p>

          </div>

          <Link
            to="/analyze"
            className="
              inline-flex
              items-center
              justify-center
              gap-2
              bg-blue-600
              hover:bg-blue-500
              text-white
              font-semibold
              px-5
              py-3
              rounded-xl
              transition
              shadow-lg
              shadow-blue-600/20
              whitespace-nowrap
            "
          >
            <span className="text-xl leading-none">+</span>
            New MRI Analysis
          </Link>

        </div>

        {/* =====================================================
            ERROR
        ====================================================== */}

        {error && (
          <div className="mb-6">
            <ErrorAlert message={error} />
          </div>
        )}

        {/* =====================================================
            STAT CARDS
        ====================================================== */}

        <div className="
          grid
          grid-cols-1
          md:grid-cols-3
          gap-4
          mb-6
          w-full
        ">

          <StatCard
            title="Total Analyses"
            value={totalAnalyses}
            subtitle="MRI scans analyzed"
            icon="🧠"
          />

          <StatCard
            title="Reports Generated"
            value={reports}
            subtitle="Available results"
            icon="📄"
          />

          <StatCard
            title="Tumor Detected"
            value={tumorDetected}
            subtitle="AI-detected tumor cases"
            icon="⚠️"
            danger
          />

        </div>

        {/* =====================================================
            LATEST ANALYSIS
        ====================================================== */}

        <div className="
          bg-[#06152d]
          border
          border-blue-900/60
          rounded-2xl
          overflow-hidden
          w-full
          min-w-0
          mb-6
        ">

          {/* HEADER */}

          <div className="
            flex
            flex-col
            sm:flex-row
            sm:items-center
            sm:justify-between
            gap-3
            p-5
            sm:p-6
            border-b
            border-blue-900/50
          ">

            <div>

              <h2 className="
                text-xl
                font-semibold
                text-white
              ">
                Latest Analysis
              </h2>

              <p className="
                text-sm
                text-slate-400
                mt-1
              ">
                Most recent MRI result
              </p>

            </div>

            <Link
              to="/history"
              className="
                text-blue-400
                hover:text-blue-300
                text-sm
                font-medium
                whitespace-nowrap
              "
            >
              View history →
            </Link>

          </div>

          {/* =================================================
              LATEST RESULT
          ================================================== */}

          {latest ? (

            <div className="p-5 sm:p-6">

              {/* MRI INFORMATION */}

              <div className="
                flex
                flex-col
                sm:flex-row
                sm:items-center
                gap-4
                mb-6
                min-w-0
              ">

                <div className="
                  w-16
                  h-16
                  shrink-0
                  rounded-2xl
                  bg-blue-500/10
                  border
                  border-blue-500/20
                  flex
                  items-center
                  justify-center
                  text-3xl
                ">
                  🧠
                </div>

                <div className="
                  flex-1
                  min-w-0
                ">

                  <p className="
                    text-xs
                    text-blue-400
                    uppercase
                    tracking-wide
                  ">
                    MRI Scan
                  </p>

                  <h3 className="
                    text-lg
                    font-semibold
                    text-white
                    mt-1
                    truncate
                  ">
                    {latest.original_filename || 'MRI Image'}
                  </h3>

                  <p className="
                    text-sm
                    text-slate-400
                    mt-1
                  ">
                    {latest.created_at
                      ? new Date(latest.created_at).toLocaleString()
                      : 'Date unavailable'}
                  </p>

                </div>

                <div className="shrink-0">
                  <StatusBadge
                    status={latest.status || 'completed'}
                  />
                </div>

              </div>

              {/* =================================================
                  RESULT CARDS
              ================================================== */}

              <div className="
                grid
                grid-cols-1
                md:grid-cols-3
                gap-4
                mb-6
              ">

                {/* DETECTION */}

                <div className="
                  bg-[#081b36]
                  rounded-xl
                  p-4
                  border
                  border-blue-900/60
                ">

                  <p className="text-sm text-slate-400">
                    Detection
                  </p>

                  <p
                    className={`
                      text-xl
                      font-bold
                      mt-2
                      ${
                        latest.predicted_class &&
                        latest.predicted_class.toLowerCase() !== 'no tumor'
                          ? 'text-red-400'
                          : 'text-emerald-400'
                      }
                    `}
                  >
                    {latest.predicted_class &&
                    latest.predicted_class.toLowerCase() !== 'no tumor'
                      ? 'Detected'
                      : 'Not Detected'}
                  </p>

                </div>

                {/* CLASSIFICATION */}

                <div className="
                  bg-[#081b36]
                  rounded-xl
                  p-4
                  border
                  border-blue-900/60
                ">

                  <p className="text-sm text-slate-400">
                    Classification
                  </p>

                  <p className="
                    text-xl
                    font-bold
                    text-white
                    mt-2
                    uppercase
                    truncate
                  ">
                    {latest.predicted_class || 'N/A'}
                  </p>

                </div>

                {/* CONFIDENCE */}

                <div className="
                  bg-[#081b36]
                  rounded-xl
                  p-4
                  border
                  border-blue-900/60
                ">

                  <p className="text-sm text-slate-400">
                    Confidence
                  </p>

                  <div className="mt-2">

                    {latest.confidence != null ? (

                      <ConfidenceBadge
                        level={latest.confidence_level}
                        confidence={latest.confidence}
                      />

                    ) : (

                      <span className="text-slate-400">
                        N/A
                      </span>

                    )}

                  </div>

                </div>

              </div>

              {/* VIEW RESULT */}

              <div className="flex flex-wrap gap-3">

                <Link
                  to={`/result/${latest.id}`}
                  className="
                    inline-flex
                    items-center
                    justify-center
                    bg-blue-600
                    hover:bg-blue-500
                    text-white
                    font-semibold
                    px-5
                    py-3
                    rounded-xl
                    transition
                  "
                >
                  View Full Result
                </Link>

              </div>

            </div>

          ) : (

            /* =================================================
                NO ANALYSIS
            ================================================== */

            <div className="p-10 text-center">

              <div className="text-5xl mb-4">
                🧠
              </div>

              <h3 className="
                text-lg
                font-semibold
                text-white
              ">
                No MRI analysis yet
              </h3>

              <p className="
                text-sm
                text-slate-400
                mt-2
              ">
                Upload your first brain MRI to begin an AI-assisted analysis.
              </p>

              <Link
                to="/analyze"
                className="
                  inline-flex
                  mt-5
                  bg-blue-600
                  hover:bg-blue-500
                  text-white
                  font-semibold
                  px-5
                  py-3
                  rounded-xl
                "
              >
                Start MRI Analysis
              </Link>

            </div>

          )}

        </div>

        {/* =====================================================
            RECENT ANALYSIS
        ====================================================== */}

        <div className="
          bg-[#06152d]
          border
          border-blue-900/60
          rounded-2xl
          overflow-hidden
          w-full
          min-w-0
        ">

          <div className="
            p-5
            sm:p-6
            border-b
            border-blue-900/50
          ">

            <h2 className="
              text-xl
              font-semibold
              text-white
            ">
              Recent Analysis
            </h2>

            <p className="
              text-sm
              text-slate-400
              mt-1
            ">
              Your latest MRI analysis records
            </p>

          </div>

          {history.length === 0 ? (

            <div className="
              p-10
              text-center
              text-slate-400
            ">
              No analyses available.
            </div>

          ) : (

            <div className="w-full overflow-x-auto">

              <table className="
                w-full
                min-w-[850px]
                text-sm
              ">

                <thead>

                  <tr className="
                    text-left
                    text-slate-400
                    border-b
                    border-blue-900/50
                  ">

                    <th className="
                      px-5
                      sm:px-6
                      py-4
                      whitespace-nowrap
                      font-medium
                    ">
                      Date
                    </th>

                    <th className="
                      px-5
                      sm:px-6
                      py-4
                      whitespace-nowrap
                      font-medium
                    ">
                      Image Name
                    </th>

                    <th className="
                      px-5
                      sm:px-6
                      py-4
                      whitespace-nowrap
                      font-medium
                    ">
                      Result
                    </th>

                    <th className="
                      px-5
                      sm:px-6
                      py-4
                      whitespace-nowrap
                      font-medium
                    ">
                      Confidence
                    </th>

                    <th className="
                      px-5
                      sm:px-6
                      py-4
                      whitespace-nowrap
                      font-medium
                    ">
                      Status
                    </th>

                    <th className="
                      px-5
                      sm:px-6
                      py-4
                      whitespace-nowrap
                      font-medium
                      text-right
                    ">
                      Action
                    </th>

                  </tr>

                </thead>

                <tbody>

                  {history.map((item) => (

                    <tr
                      key={item.id}
                      className="
                        border-b
                        border-blue-900/40
                        hover:bg-blue-900/20
                        transition
                      "
                    >

                      <td className="
                        px-5
                        sm:px-6
                        py-5
                        text-slate-300
                        whitespace-nowrap
                      ">
                        {item.created_at
                          ? new Date(item.created_at).toLocaleDateString()
                          : '—'}
                      </td>

                      <td className="
                        px-5
                        sm:px-6
                        py-5
                        text-slate-300
                        max-w-[280px]
                      ">

                        <div
                          className="
                            truncate
                            max-w-[250px]
                          "
                          title={
                            item.original_filename ||
                            'MRI Image'
                          }
                        >
                          {item.original_filename ||
                            'MRI Image'}
                        </div>

                      </td>

                      <td className="
                        px-5
                        sm:px-6
                        py-5
                      ">

                        <span className="
                          font-medium
                          text-white
                          whitespace-nowrap
                        ">
                          {item.predicted_class || '—'}
                        </span>

                      </td>

                      <td className="
                        px-5
                        sm:px-6
                        py-5
                      ">

                        {item.confidence != null ? (

                          <ConfidenceBadge
                            level={item.confidence_level}
                            confidence={item.confidence}
                          />

                        ) : (

                          <span className="text-slate-400">
                            —
                          </span>

                        )}

                      </td>

                      <td className="
                        px-5
                        sm:px-6
                        py-5
                      ">

                        <StatusBadge
                          status={item.status || 'completed'}
                        />

                      </td>

                      <td className="
                        px-5
                        sm:px-6
                        py-5
                        text-right
                      ">

                        <Link
                          to={`/result/${item.id}`}
                          className="
                            inline-flex
                            items-center
                            justify-center
                            gap-1
                            text-blue-400
                            hover:text-blue-300
                            font-semibold
                            whitespace-nowrap
                          "
                        >
                          View
                          <span>→</span>
                        </Link>

                      </td>

                    </tr>

                  ))}

                </tbody>

              </table>

            </div>

          )}

        </div>

        {/* =====================================================
            IMPORTANT NOTICE
        ====================================================== */}

        <div className="
          mt-6
          rounded-xl
          border
          border-blue-500/20
          bg-blue-500/5
          px-5
          py-4
          flex
          items-start
          gap-3
          w-full
        ">

          <span className="text-xl shrink-0">
            🛡️
          </span>

          <p className="
            text-sm
            text-slate-400
            leading-6
          ">

            <span className="
              font-semibold
              text-blue-400
            ">
              Important:
            </span>{' '}

            AI output is decision-support and does not replace
            professional medical diagnosis.

          </p>

        </div>

      </main>

    </div>
  )
}


/* =====================================================
   STAT CARD
===================================================== */

function StatCard({
  title,
  value,
  subtitle,
  icon,
  danger = false,
}) {
  return (
    <div className="
      bg-[#06152d]
      border
      border-blue-900/60
      rounded-2xl
      p-5
      hover:border-blue-700/60
      transition
      w-full
      min-w-0
    ">

      <div className="
        flex
        items-start
        justify-between
        gap-4
      ">

        <div className="min-w-0">

          <p className="text-sm text-slate-400">
            {title}
          </p>

          <p
            className={`
              text-2xl
              font-bold
              mt-3
              ${
                danger
                  ? 'text-red-400'
                  : 'text-white'
              }
            `}
          >
            {value}
          </p>

          <p className="
            text-xs
            text-slate-500
            mt-2
          ">
            {subtitle}
          </p>

        </div>

        <div className="
          w-12
          h-12
          shrink-0
          rounded-xl
          bg-blue-900/30
          border
          border-blue-800/50
          flex
          items-center
          justify-center
          text-xl
        ">
          {icon}
        </div>

      </div>

    </div>
  )
}