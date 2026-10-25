import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'
import { ErrorAlert } from '../components/Common.jsx'

const ALLOWED_TYPES = [
  'image/jpeg',
  'image/jpg',
  'image/png',
]

const MAX_SIZE_MB = 10

const STAGES = [
  'Uploading image',
  'Validating brain MRI image',
  'Checking image quality',
  'Preprocessing',
  'Running tumor classification',
  'Generating Grad-CAM explanation',
  'Finalizing result',
]

export default function Analyze() {
  const navigate = useNavigate()

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [error, setError] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [stageIdx, setStageIdx] = useState(0)

  const inputRef = useRef(null)

  // =========================================================
  // FILE VALIDATION
  // =========================================================

  const validateAndSet = (f) => {
    setError('')

    if (!f) {
      return
    }

    // Check file type
    if (!ALLOWED_TYPES.includes(f.type)) {
      setError(
        'Unsupported format. Please upload a JPG, JPEG, or PNG image.'
      )
      return
    }

    // Check empty file
    if (f.size === 0) {
      setError('The selected file is empty.')
      return
    }

    // Check file size
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(
        `File exceeds the maximum allowed size of ${MAX_SIZE_MB} MB.`
      )
      return
    }

    setFile(f)
    setPreview(URL.createObjectURL(f))
  }

  // =========================================================
  // DRAG & DROP
  // =========================================================

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)

    validateAndSet(
      e.dataTransfer.files?.[0]
    )
  }

  // =========================================================
  // REMOVE FILE
  // =========================================================

  const removeFile = () => {
    setFile(null)
    setPreview(null)
    setError('')

    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  // =========================================================
  // ANALYZE MRI
  // =========================================================

  const handleAnalyze = async () => {
    if (!file) {
      return
    }

    setLoading(true)
    setError('')
    setStageIdx(0)

    const stageTimer = setInterval(() => {
      setStageIdx((i) =>
        Math.min(
          i + 1,
          STAGES.length - 2
        )
      )
    }, 900)

    try {
      const formData = new FormData()

      formData.append(
        'file',
        file
      )

      const res = await api.post(
        '/api/analysis/upload',
        formData,
        {
          headers: {
            'Content-Type':
              'multipart/form-data',
          },
        }
      )

      clearInterval(stageTimer)

      setStageIdx(
        STAGES.length - 1
      )

      setTimeout(() => {
        navigate(
          `/result/${res.data.id}`
        )
      }, 400)

    } catch (err) {
      clearInterval(stageTimer)

      const detail =
        err.response?.data?.detail

      if (
        typeof detail === 'string'
      ) {
        setError(detail)
      } else {
        setError(
          'Analysis failed. Please try again.'
        )
      }

      setLoading(false)
    }
  }

  // =========================================================
  // PAGE
  // =========================================================

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

        {/* =====================================================
            PAGE HEADER
        ====================================================== */}

        <div className="mb-8">

          <h1 className="
            text-3xl
            sm:text-4xl
            font-bold
            text-white
          ">
            New MRI Analysis
          </h1>

          <p className="
            text-slate-400
            text-sm
            sm:text-base
            mt-2
          ">
            Upload a brain MRI image (JPG, JPEG, or PNG).
          </p>

        </div>

        {/* =====================================================
            ERROR MESSAGE
        ====================================================== */}

        {error && (
          <div className="mb-6">
            <ErrorAlert message={error} />
          </div>
        )}

        {/* =====================================================
            NO FILE SELECTED
        ====================================================== */}

        {!file ? (

          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragActive(true)
            }}

            onDragLeave={() => {
              setDragActive(false)
            }}

            onDrop={handleDrop}

            onClick={() => {
              inputRef.current?.click()
            }}

            className={`
              bg-[#06152d]
              rounded-2xl
              border-2
              border-dashed
              p-10
              sm:p-14
              text-center
              cursor-pointer
              transition
              shadow-xl
              shadow-blue-950/20

              ${
                dragActive
                  ? `
                    border-blue-500
                    bg-blue-900/20
                  `
                  : `
                    border-blue-900/70
                    hover:border-blue-700
                    hover:bg-[#081b36]
                  `
              }
            `}
          >

            {/* BRAIN ICON */}

            <div className="
              w-20
              h-20
              mx-auto
              rounded-2xl
              bg-blue-500/10
              border
              border-blue-500/20
              flex
              items-center
              justify-center
              text-5xl
              mb-5
            ">
              🧠
            </div>

            {/* MAIN TEXT */}

            <p className="
              font-semibold
              text-white
              text-lg
              sm:text-xl
            ">
              Drag & drop your MRI image here
            </p>

            {/* SUBTEXT */}

            <p className="
              text-sm
              text-slate-400
              mt-2
            ">
              or click to browse
            </p>

            {/* FORMAT INFO */}

            <div className="
              mt-4
              inline-flex
              flex-wrap
              justify-center
              items-center
              gap-2
            ">

              <span className="
                px-3
                py-1.5
                rounded-lg
                bg-blue-500/10
                border
                border-blue-500/20
                text-xs
                text-blue-300
              ">
                JPG
              </span>

              <span className="
                px-3
                py-1.5
                rounded-lg
                bg-blue-500/10
                border
                border-blue-500/20
                text-xs
                text-blue-300
              ">
                JPEG
              </span>

              <span className="
                px-3
                py-1.5
                rounded-lg
                bg-blue-500/10
                border
                border-blue-500/20
                text-xs
                text-blue-300
              ">
                PNG
              </span>

              <span className="
                px-3
                py-1.5
                rounded-lg
                bg-slate-800
                border
                border-slate-700
                text-xs
                text-slate-400
              ">
                Max {MAX_SIZE_MB} MB
              </span>

            </div>

            {/* HIDDEN FILE INPUT */}

            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              className="hidden"

              onChange={(e) => {
                validateAndSet(
                  e.target.files?.[0]
                )
              }}
            />

          </div>

        ) : (

          /* ===================================================
             FILE SELECTED
          ==================================================== */

          <div className="
            bg-[#06152d]
            border
            border-blue-900/60
            rounded-2xl
            p-5
            sm:p-6
            shadow-xl
            shadow-blue-950/20
            space-y-6
          ">

            {/* FILE PREVIEW */}

            <div className="
              flex
              flex-col
              sm:flex-row
              gap-5
              items-start
            ">

              <img
                src={preview}
                alt="MRI preview"
                className="
                  w-full
                  sm:w-36
                  h-36
                  object-cover
                  rounded-xl
                  border
                  border-blue-900/60
                  bg-slate-950
                "
              />

              <div className="
                flex-1
                min-w-0
              ">

                <p className="
                  font-semibold
                  text-white
                  truncate
                ">
                  {file.name}
                </p>

                <p className="
                  text-sm
                  text-slate-400
                  mt-2
                ">
                  {(file.size / 1024).toFixed(1)} KB
                </p>

                {!loading && (
                  <button
                    type="button"
                    onClick={removeFile}
                    className="
                      text-sm
                      text-red-400
                      hover:text-red-300
                      mt-3
                      transition
                    "
                  >
                    Remove image
                  </button>
                )}

              </div>

            </div>

            {/* =================================================
                ANALYSIS STAGES
            ================================================== */}

            {loading && (

              <div className="
                bg-[#081b36]
                border
                border-blue-900/60
                rounded-xl
                p-5
              ">

                <h2 className="
                  font-semibold
                  text-white
                  mb-4
                ">
                  MRI Analysis Progress
                </h2>

                <div className="space-y-3">

                  {STAGES.map((stage, i) => (

                    <div
                      key={stage}
                      className="
                        flex
                        items-center
                        gap-3
                        text-sm
                      "
                    >

                      <span className="
                        w-6
                        text-center
                      ">
                        {i < stageIdx
                          ? '✅'
                          : i === stageIdx
                            ? '⏳'
                            : '⚪'}
                      </span>

                      <span
                        className={
                          i <= stageIdx
                            ? `
                              text-slate-100
                              font-medium
                            `
                            : `
                              text-slate-500
                            `
                        }
                      >
                        {stage}
                      </span>

                    </div>

                  ))}

                </div>

              </div>

            )}

            {/* ERROR */}

            <ErrorAlert message={error} />

            {/* ANALYZE BUTTON */}

            <button
              type="button"
              onClick={handleAnalyze}
              disabled={loading}
              className={`
                w-full
                py-3.5
                rounded-xl
                font-semibold
                text-white
                transition
                ${
                  loading
                    ? `
                      bg-blue-900
                      cursor-not-allowed
                      opacity-70
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
              {loading
                ? 'Analyzing…'
                : 'Analyze MRI'}
            </button>

          </div>

        )}

      </main>

    </div>
  )
}