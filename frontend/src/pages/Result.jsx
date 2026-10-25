import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import api, { fetchImageBlobUrl } from '../services/api'
import {
  ConfidenceBadge,
  Spinner,
  ErrorAlert,
} from '../components/Common.jsx'

const VIEW_TABS = [
  { key: 'original', label: 'Original' },
  { key: 'gradcam', label: 'Grad-CAM' },
  { key: 'segmentation', label: 'Segmentation' },
]

// =====================================================
// DISPLAY CLASS NAME
// =====================================================

function formatClassName(className) {
  if (!className) return 'N/A'

  if (className.toLowerCase() === 'notumor') {
    return 'No tumor'
  }

  return className
}

// =====================================================
// RESULT PAGE
// =====================================================

export default function Result() {
  const { id } = useParams()

  const [analysis, setAnalysis] = useState(null)
  const [images, setImages] = useState({})
  const [activeTab, setActiveTab] = useState('original')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [reportBusy, setReportBusy] = useState(false)
  const [reportMsg, setReportMsg] = useState('')

  // ===================================================
  // LOAD ANALYSIS
  // ===================================================

  useEffect(() => {
    let cancelled = false

    ;(async () => {
      try {
        setLoading(true)
        setError('')

        const res = await api.get(`/api/analysis/${id}`)

        if (cancelled) return

        setAnalysis(res.data)

        const imgTypes = ['original']

        if (res.data.status === 'completed') {
          imgTypes.push('gradcam')

          if (res.data.segmentation_available) {
            imgTypes.push('segmentation')
          }
        }

        const entries = await Promise.all(
          imgTypes.map(async (type) => {
            try {
              const url = await fetchImageBlobUrl(
                `/api/analysis/${id}/image/${type}`
              )

              return [type, url]
            } catch (err) {
              console.warn(
                `Unable to load ${type} image:`,
                err
              )

              return [type, null]
            }
          })
        )

        if (!cancelled) {
          setImages(
            Object.fromEntries(entries)
          )
        }

      } catch (err) {
        if (!cancelled) {
          setError(
            err.response?.data?.detail ||
              'Failed to load analysis.'
          )
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [id])

  // ===================================================
  // GENERATE REPORT
  // ===================================================

  const handleGenerateReport = async () => {
    setReportBusy(true)
    setReportMsg('')

    try {
      await api.post(
        `/api/reports/generate/${id}`
      )

      const res = await api.get(
        `/api/reports/${id}`,
        {
          responseType: 'blob',
        }
      )

      const url = URL.createObjectURL(
        res.data
      )

      const a = document.createElement('a')

      a.href = url
      a.download = `brain_mri_report_${id}.pdf`

      document.body.appendChild(a)
      a.click()
      a.remove()

      URL.revokeObjectURL(url)

      setReportMsg(
        'Report generated and downloaded.'
      )

    } catch (err) {
      setReportMsg(
        err.response?.data?.detail ||
          'Report generation failed.'
      )
    } finally {
      setReportBusy(false)
    }
  }

  // ===================================================
  // LOADING
  // ===================================================

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

  // ===================================================
  // NO ANALYSIS
  // ===================================================

  if (!analysis) {
    return (
      <div className="
        min-h-screen
        bg-[#020b1c]
        p-6
      ">
        <ErrorAlert
          message={
            error ||
            'Analysis not found.'
          }
        />
      </div>
    )
  }

  // ===================================================
  // CLASS PROBABILITIES
  // ===================================================

  const probs =
    analysis.class_probabilities
      ? Object.entries(
          analysis.class_probabilities
        ).sort(
          (a, b) => b[1] - a[1]
        )
      : []

  // ===================================================
  // ACTIVE IMAGE
  // ===================================================

  const activeImage =
    images[activeTab]

  // ===================================================
  // RESULT PAGE
  // ===================================================

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
        max-w-5xl
        mx-auto
        p-4
        sm:p-6
        lg:p-8
      ">

        {/* =================================================
            HEADER
        ================================================== */}

        <div className="
          flex
          flex-col
          sm:flex-row
          sm:items-center
          sm:justify-between
          gap-4
          mb-8
        ">

          <div className="min-w-0">

            <h1 className="
              text-3xl
              sm:text-4xl
              font-bold
              text-white
            ">
              Brain MRI Analysis
            </h1>

            <p className="
              text-slate-400
              text-sm
              mt-2
              truncate
              max-w-[600px]
            ">
              {analysis.original_filename ||
                'MRI Image'}
            </p>

          </div>

          <Link
            to="/analyze"
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
              shadow-lg
              shadow-blue-600/20
              whitespace-nowrap
            "
          >
            New Analysis
          </Link>

        </div>

        {/* =================================================
            FAILED ANALYSIS
        ================================================== */}

        {analysis.status !== 'completed' ? (

          <div className="
            bg-[#06152d]
            border
            border-red-500/20
            rounded-2xl
            p-6
          ">

            <ErrorAlert
              message={
                analysis.failure_reason ||
                'This analysis did not complete.'
              }
            />

          </div>

        ) : (

          <>

            {/* =============================================
                MRI / GRAD-CAM VIEWER
            ============================================== */}

            <div className="
              bg-[#06152d]
              border
              border-blue-900/60
              rounded-2xl
              p-4
              sm:p-6
              shadow-xl
              shadow-blue-950/20
              mb-6
            ">

              {/* VIEW TABS */}

              <div className="
                flex
                flex-wrap
                gap-2
                mb-5
              ">

                {VIEW_TABS
                  .filter((tab) => {
                    if (tab.key === 'segmentation') {
                      return Boolean(
                        analysis.segmentation_available
                      )
                    }

                    return true
                  })
                  .map((tab) => {

                    const hasImage =
                      Boolean(
                        images[tab.key]
                      )

                    return (
                      <button
                        key={tab.key}
                        type="button"
                        onClick={() =>
                          setActiveTab(
                            tab.key
                          )
                        }
                        className={`
                          px-4
                          py-2
                          rounded-lg
                          text-sm
                          font-semibold
                          transition
                          border

                          ${
                            activeTab === tab.key
                              ? `
                                bg-blue-600
                                text-white
                                border-blue-500
                                shadow-lg
                                shadow-blue-600/20
                              `
                              : hasImage
                                ? `
                                  bg-[#081b36]
                                  text-slate-300
                                  border-blue-900/60
                                  hover:bg-blue-900/30
                                  hover:text-white
                                `
                                : `
                                  bg-[#081b36]
                                  text-slate-600
                                  border-blue-900/40
                                `
                          }
                        `}
                      >
                        {tab.label}
                      </button>
                    )
                  })}

              </div>

              {/* IMAGE VIEWER */}

              <div className="
                bg-[#020b1c]
                border
                border-blue-900/50
                rounded-2xl
                p-4
                sm:p-6
                min-h-[420px]
                flex
                items-center
                justify-center
              ">

                {activeImage ? (

                  <img
                    src={activeImage}
                    alt={activeTab}
                    className="
                      block
                      w-auto
                      max-w-full
                      max-h-[480px]
                      mx-auto
                      rounded-xl
                      border
                      border-blue-900/60
                      object-contain
                    "
                  />

                ) : (

                  <div className="
                    text-center
                    py-16
                    px-6
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
                      {activeTab === 'gradcam'
                        ? 'Grad-CAM image is not available.'
                        : activeTab === 'segmentation'
                          ? 'Segmentation image is not available.'
                          : 'Image not available.'}
                    </p>

                    {activeTab === 'gradcam' && (
                      <p className="
                        text-sm
                        text-slate-500
                        mt-2
                      ">
                        The backend must generate and
                        return the Grad-CAM image.
                      </p>
                    )}

                  </div>

                )}

              </div>

              {/* ORIGINAL DESCRIPTION */}

              {activeTab === 'original' && (

                <p className="
                  text-xs
                  text-slate-500
                  mt-4
                  text-center
                ">
                  Original brain MRI image used for AI analysis.
                </p>

              )}

              {/* GRAD-CAM DESCRIPTION */}

              {activeTab === 'gradcam' && (

                <div className="
                  mt-4
                  bg-blue-500/5
                  border
                  border-blue-500/20
                  rounded-xl
                  p-4
                ">

                  <p className="
                    text-xs
                    text-slate-400
                    leading-6
                    text-center
                  ">
                    <span className="
                      text-blue-400
                      font-semibold
                    ">
                      Grad-CAM:
                    </span>{' '}
                    highlighted regions represent image
                    areas that contributed strongly to the
                    model's prediction. This visualization
                    supports interpretation but does not
                    prove the presence or absence of a tumor.
                  </p>

                </div>

              )}

              {/* SEGMENTATION DESCRIPTION */}

              {activeTab === 'segmentation' && (

                <div className="
                  mt-4
                  bg-blue-500/5
                  border
                  border-blue-500/20
                  rounded-xl
                  p-4
                ">

                  <p className="
                    text-xs
                    text-slate-400
                    text-center
                  ">
                    Detected tumor region
                    (model-estimated).
                  </p>

                </div>

              )}

            </div>

            {/* =============================================
                RESULT SUMMARY
            ============================================== */}

            <div className="
              grid
              grid-cols-1
              md:grid-cols-3
              gap-4
              mb-6
            ">

              {/* TUMOR DETECTION */}

              <ResultStat
                label="Tumor Detection"
                value={
                  analysis.tumor_detected
                    ? 'DETECTED'
                    : 'NOT DETECTED'
                }
                emphasis={
                  analysis.tumor_detected
                    ? 'text-red-400'
                    : 'text-emerald-400'
                }
              />

              {/* CLASSIFICATION */}

              <ResultStat
                label="Classification"
                value={formatClassName(
                  analysis.predicted_class
                )}
              />

              {/* CONFIDENCE */}

              <div className="
                bg-[#081b36]
                border
                border-blue-900/60
                rounded-xl
                p-5
              ">

                <div className="
                  text-xs
                  text-slate-400
                  mb-2
                ">
                  Confidence
                </div>

                <ConfidenceBadge
                  level={
                    analysis.confidence_level
                  }
                  confidence={
                    analysis.confidence
                  }
                />

              </div>

            </div>

            {/* =============================================
                LOW CONFIDENCE
            ============================================== */}

            {analysis.confidence_level === 'low' && (

              <div className="
                bg-amber-500/10
                border
                border-amber-500/20
                rounded-2xl
                p-5
                mb-6
              ">

                <h3 className="
                  text-sm
                  font-semibold
                  text-amber-300
                  mb-2
                ">
                  Low confidence result
                </h3>

                <p className="
                  text-sm
                  text-amber-200/80
                  leading-6
                ">
                  The model confidence is relatively low.
                  Interpret the result carefully.
                </p>

              </div>

            )}

            {/* =============================================
                CLASS PROBABILITIES
            ============================================== */}

            {probs.length > 0 && (

              <div className="
                bg-[#06152d]
                border
                border-blue-900/60
                rounded-2xl
                p-5
                sm:p-6
                shadow-xl
                shadow-blue-950/20
                mb-6
              ">

                <h3 className="
                  text-lg
                  font-semibold
                  text-white
                  mb-5
                ">
                  Class Probabilities
                </h3>

                <div className="space-y-4">

                  {probs.map(
                    ([cls, probability]) => (

                      <div
                        key={cls}
                        className="
                          flex
                          items-center
                          gap-3
                          text-sm
                        "
                      >

                        <span className="
                          w-28
                          sm:w-36
                          truncate
                          text-slate-300
                        ">
                          {formatClassName(cls)}
                        </span>

                        <div className="
                          flex-1
                          h-2.5
                          bg-[#081b36]
                          rounded-full
                          overflow-hidden
                          border
                          border-blue-900/40
                        ">

                          <div
                            className="
                              h-full
                              bg-blue-500
                              rounded-full
                            "
                            style={{
                              width: `${probability * 100}%`,
                            }}
                          />

                        </div>

                        <span className="
                          w-16
                          text-right
                          font-semibold
                          text-white
                        ">
                          {(probability * 100).toFixed(1)}%
                        </span>

                      </div>

                    )
                  )}

                </div>

              </div>

            )}

            {/* =============================================
                MODEL INFORMATION
            ============================================== */}

            <div className="
              bg-[#06152d]
              border
              border-blue-900/60
              rounded-2xl
              p-5
              sm:p-6
              shadow-xl
              shadow-blue-950/20
              mb-6
            ">

              <p className="
                text-sm
                text-slate-400
              ">

                <span className="
                  font-semibold
                  text-white
                ">
                  Model:
                </span>{' '}

                <span className="text-blue-300">
                  {analysis.model_name ||
                    'Unknown model'}
                </span>{' '}

                <span className="text-slate-500">
                  (
                  {analysis.model_version ||
                    'Unknown version'}
                  )
                </span>

              </p>

            </div>

            {/* =============================================
                ACTION BUTTONS
            ============================================== */}

            <div className="
              flex
              flex-wrap
              gap-3
              mb-5
            ">

              <button
                type="button"
                onClick={
                  handleGenerateReport
                }
                disabled={reportBusy}
                className={`
                  inline-flex
                  items-center
                  justify-center
                  px-5
                  py-3
                  rounded-xl
                  font-semibold
                  text-white
                  transition

                  ${
                    reportBusy
                      ? `
                        bg-blue-900
                        opacity-70
                        cursor-not-allowed
                      `
                      : `
                        bg-blue-600
                        hover:bg-blue-500
                        shadow-lg
                        shadow-blue-600/20
                      `
                  }
                `}
              >
                {reportBusy
                  ? 'Generating…'
                  : 'Download Report'}
              </button>

              <Link
                to="/history"
                className="
                  inline-flex
                  items-center
                  justify-center
                  px-5
                  py-3
                  rounded-xl
                  font-semibold
                  bg-[#081b36]
                  border
                  border-blue-900/60
                  text-slate-200
                  hover:bg-blue-900/30
                  hover:text-white
                  transition
                "
              >
                View History
              </Link>

              <Link
                to="/analyze"
                className="
                  inline-flex
                  items-center
                  justify-center
                  px-5
                  py-3
                  rounded-xl
                  font-semibold
                  bg-[#081b36]
                  border
                  border-blue-900/60
                  text-slate-200
                  hover:bg-blue-900/30
                  hover:text-white
                  transition
                "
              >
                New Analysis
              </Link>

            </div>

            {/* =============================================
                REPORT MESSAGE
            ============================================== */}

            {reportMsg && (

              <div className="
                bg-blue-500/5
                border
                border-blue-500/20
                rounded-xl
                px-4
                py-3
                mb-6
              ">

                <p className="
                  text-sm
                  text-slate-300
                ">
                  {reportMsg}
                </p>

              </div>

            )}

          </>

        )}

        {/* =================================================
            IMPORTANT NOTICE
        ================================================== */}

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
        ">

          <span className="
            text-xl
            shrink-0
          ">
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


// =====================================================
// RESULT STAT COMPONENT
// =====================================================

function ResultStat({
  label,
  value,
  emphasis = '',
}) {
  return (
    <div className="
      bg-[#081b36]
      border
      border-blue-900/60
      rounded-xl
      p-5
    ">

      <div className="
        text-xs
        text-slate-400
        mb-2
      ">
        {label}
      </div>

      <div
        className={`
          font-bold
          text-xl
          ${
            emphasis ||
            'text-white'
          }
        `}
      >
        {value}
      </div>

    </div>
  )
}