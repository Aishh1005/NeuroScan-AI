import React, { useEffect, useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import api from '../services/api'
import {
  Spinner,
  ErrorAlert,
  StatusBadge,
  ConfidenceBadge,
} from '../components/Common.jsx'

export default function History() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [sortDesc, setSortDesc] = useState(true)

  useEffect(() => {
    loadHistory()
  }, [])

  async function loadHistory() {
    try {
      setLoading(true)
      setError('')

      const res = await api.get('/api/analysis/history')

      setItems(res.data || [])
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Failed to load history.'
      )
    } finally {
      setLoading(false)
    }
  }

  const filtered = useMemo(() => {
    let out = items.filter((item) => {
      const filename =
        item.original_filename || ''

      const prediction =
        item.predicted_class || ''

      const query =
        search.toLowerCase().trim()

      return (
        filename.toLowerCase().includes(query) ||
        prediction.toLowerCase().includes(query)
      )
    })

    if (statusFilter !== 'all') {
      out = out.filter(
        (item) =>
          item.status === statusFilter
      )
    }

    out.sort((a, b) =>
      sortDesc
        ? new Date(b.created_at) -
          new Date(a.created_at)
        : new Date(a.created_at) -
          new Date(b.created_at)
    )

    return out
  }, [
    items,
    search,
    statusFilter,
    sortDesc,
  ])

  if (loading) {
    return (
      <div className="
        min-h-screen
        bg-[#020b1c]
        flex
        items-center
        justify-center
      ">
        <Spinner />
      </div>
    )
  }

  return (
    <div className="
      min-h-screen
      w-full
      bg-[#020b1c]
      text-white
      overflow-x-hidden
    ">

      <main className="
        w-full
        p-4
        sm:p-6
        lg:p-8
      ">

        {/* =====================================================
            HEADER
        ====================================================== */}

        <div className="mb-8">

          <h1 className="
            text-3xl
            sm:text-4xl
            font-bold
            text-white
          ">
            Analysis History
          </h1>

          <p className="
            text-slate-400
            text-sm
            sm:text-base
            mt-2
          ">
            All MRI analyses linked to your account.
          </p>

        </div>

        {/* =====================================================
            ERROR
        ====================================================== */}

        <div className="mb-6">
          <ErrorAlert message={error} />
        </div>

        {/* =====================================================
            FILTERS
        ====================================================== */}

        <div className="
          grid
          grid-cols-1
          lg:grid-cols-12
          gap-4
          mb-6
        ">

          {/* SEARCH */}

          <div className="lg:col-span-6">

            <label className="
              block
              text-sm
              font-medium
              text-slate-300
              mb-2
            ">
              Search
            </label>

            <input
              type="text"
              placeholder="Search by filename or class..."
              value={search}
              onChange={(e) =>
                setSearch(e.target.value)
              }
              className="
                w-full
                bg-[#06152d]
                border
                border-blue-900/60
                text-white
                placeholder:text-slate-500
                rounded-xl
                px-4
                py-3
                outline-none
                focus:border-blue-500
                focus:ring-2
                focus:ring-blue-500/20
                transition
              "
            />

          </div>

          {/* STATUS */}

          <div className="lg:col-span-3">

            <label className="
              block
              text-sm
              font-medium
              text-slate-300
              mb-2
            ">
              Status
            </label>

            <select
              value={statusFilter}
              onChange={(e) =>
                setStatusFilter(e.target.value)
              }
              className="
                w-full
                bg-[#06152d]
                border
                border-blue-900/60
                text-white
                rounded-xl
                px-4
                py-3
                outline-none
                focus:border-blue-500
                focus:ring-2
                focus:ring-blue-500/20
                transition
                cursor-pointer
              "
            >

              <option
                value="all"
                className="bg-[#06152d]"
              >
                All statuses
              </option>

              <option
                value="completed"
                className="bg-[#06152d]"
              >
                Completed
              </option>

              <option
                value="failed"
                className="bg-[#06152d]"
              >
                Failed
              </option>

              <option
                value="pending"
                className="bg-[#06152d]"
              >
                Pending
              </option>

            </select>

          </div>

          {/* SORT */}

          <div className="
            lg:col-span-3
            flex
            items-end
          ">

            <button
              type="button"
              onClick={() =>
                setSortDesc((current) => !current)
              }
              className="
                w-full
                bg-blue-600
                hover:bg-blue-500
                text-white
                font-semibold
                rounded-xl
                px-4
                py-3
                transition
                shadow-lg
                shadow-blue-600/20
              "
            >
              Sort:{' '}
              {sortDesc
                ? 'Newest first'
                : 'Oldest first'}
            </button>

          </div>

        </div>

        {/* =====================================================
            RESULT COUNT
        ====================================================== */}

        <div className="
          mb-4
          text-sm
          text-slate-400
        ">
          Showing{' '}
          <span className="
            text-white
            font-semibold
          ">
            {filtered.length}
          </span>{' '}
          {filtered.length === 1
            ? 'analysis'
            : 'analyses'}
        </div>

        {/* =====================================================
            HISTORY TABLE
        ====================================================== */}

        <div className="
          bg-[#06152d]
          border
          border-blue-900/60
          rounded-2xl
          overflow-hidden
          shadow-xl
          shadow-blue-950/20
        ">

          <div className="
            overflow-x-auto
          ">

            <table className="
              w-full
              min-w-[950px]
              text-sm
            ">

              {/* TABLE HEADER */}

              <thead>

                <tr className="
                  bg-[#081b36]
                  text-left
                  text-slate-300
                  border-b
                  border-blue-900/60
                ">

                  <th className="
                    py-4
                    px-5
                    font-semibold
                    whitespace-nowrap
                  ">
                    Date
                  </th>

                  <th className="
                    py-4
                    px-5
                    font-semibold
                    whitespace-nowrap
                  ">
                    MRI Filename
                  </th>

                  <th className="
                    py-4
                    px-5
                    font-semibold
                    whitespace-nowrap
                  ">
                    Prediction
                  </th>

                  <th className="
                    py-4
                    px-5
                    font-semibold
                    whitespace-nowrap
                  ">
                    Confidence
                  </th>

                  <th className="
                    py-4
                    px-5
                    font-semibold
                    whitespace-nowrap
                  ">
                    Tumor Detected
                  </th>

                  <th className="
                    py-4
                    px-5
                    font-semibold
                    whitespace-nowrap
                  ">
                    Status
                  </th>

                  <th className="
                    py-4
                    px-5
                    font-semibold
                    text-right
                    whitespace-nowrap
                  ">
                    Action
                  </th>

                </tr>

              </thead>

              {/* TABLE BODY */}

              <tbody>

                {filtered.map((item) => (

                  <tr
                    key={item.id}
                    className="
                      border-b
                      border-blue-900/40
                      hover:bg-blue-900/20
                      transition
                    "
                  >

                    {/* DATE */}

                    <td className="
                      py-4
                      px-5
                      text-slate-300
                      whitespace-nowrap
                    ">
                      {item.created_at
                        ? new Date(
                            item.created_at
                          ).toLocaleString()
                        : '—'}
                    </td>

                    {/* FILENAME */}

                    <td className="
                      py-4
                      px-5
                      text-slate-300
                      max-w-[220px]
                    ">

                      <div
                        className="
                          truncate
                          max-w-[200px]
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

                    {/* PREDICTION */}

                    <td className="
                      py-4
                      px-5
                    ">

                      <span className="
                        font-medium
                        text-white
                        whitespace-nowrap
                      ">
                        {item.predicted_class ||
                          '—'}
                      </span>

                    </td>

                    {/* CONFIDENCE */}

                    <td className="
                      py-4
                      px-5
                    ">

                      {item.confidence != null ? (

                        <ConfidenceBadge
                          level={
                            item.confidence_level
                          }
                          confidence={
                            item.confidence
                          }
                        />

                      ) : (

                        <span className="
                          text-slate-500
                        ">
                          —
                        </span>

                      )}

                    </td>

                    {/* TUMOR DETECTED */}

                    <td className="
                      py-4
                      px-5
                    ">

                      {item.tumor_detected == null ? (

                        <span className="
                          text-slate-500
                        ">
                          —
                        </span>

                      ) : item.tumor_detected ? (

                        <span className="
                          inline-flex
                          items-center
                          px-3
                          py-1
                          rounded-full
                          bg-red-500/10
                          border
                          border-red-500/20
                          text-red-400
                          text-xs
                          font-semibold
                        ">
                          Yes
                        </span>

                      ) : (

                        <span className="
                          inline-flex
                          items-center
                          px-3
                          py-1
                          rounded-full
                          bg-emerald-500/10
                          border
                          border-emerald-500/20
                          text-emerald-400
                          text-xs
                          font-semibold
                        ">
                          No
                        </span>

                      )}

                    </td>

                    {/* STATUS */}

                    <td className="
                      py-4
                      px-5
                    ">

                      <StatusBadge
                        status={
                          item.status ||
                          'completed'
                        }
                      />

                    </td>

                    {/* ACTION */}

                    <td className="
                      py-4
                      px-5
                      text-right
                    ">

                      <Link
                        to={`/result/${item.id}`}
                        className="
                          inline-flex
                          items-center
                          gap-1
                          text-blue-400
                          hover:text-blue-300
                          font-semibold
                          whitespace-nowrap
                          transition
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

          {/* NO RESULTS */}

          {filtered.length === 0 && (

            <div className="
              p-12
              text-center
            ">

              <div className="
                w-16
                h-16
                mx-auto
                rounded-2xl
                bg-blue-500/10
                border
                border-blue-500/20
                flex
                items-center
                justify-center
                text-3xl
                mb-4
              ">
                🧠
              </div>

              <p className="
                text-slate-300
                font-medium
              ">
                No analyses match your filters.
              </p>

              <p className="
                text-sm
                text-slate-500
                mt-2
              ">
                Try changing the search or status filter.
              </p>

            </div>

          )}

        </div>

      </main>

    </div>
  )
}