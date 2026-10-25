import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth.jsx'
import { ErrorAlert } from '../components/Common.jsx'

export default function Signup() {
  const { register } = useAuth()
  const navigate = useNavigate()

  const [form, setForm] = useState({
    full_name: '',
    email: '',
    password: '',
    confirm_password: '',
    doctor_id: '',
    verification_code: '',
  })

  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  const update = (key) => (e) => {
    setForm({
      ...form,
      [key]: e.target.value,
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    setError('')
    setLoading(true)

    try {
      await register(
        form.full_name,
        form.email,
        form.password,
        form.confirm_password,
        form.doctor_id,
        form.verification_code
      )

      setSuccess(true)

      setTimeout(() => {
        navigate('/login')
      }, 2500)

    } catch (err) {
      const detail = err.response?.data?.detail

      if (Array.isArray(detail)) {
        setError(
          detail
            .map((d) => d.msg)
            .join(' ')
        )
      } else {
        setError(
          detail ||
          'Registration failed. Please check your details and try again.'
        )
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950 px-4 py-8">

      <div className="w-full max-w-md card p-8">

        {/* =========================
            HEADER
        ========================== */}

        <div className="text-center mb-6">

          <Link to="/" className="text-2xl">
            🧠
          </Link>

          <h1 className="text-xl font-bold mt-2 text-slate-900 dark:text-white">
            Create your doctor account
          </h1>

          <p className="text-sm text-slate-500 dark:text-slate-400">
            Register for NeuroScan AI
          </p>

        </div>


        {/* =========================
            SUCCESS MESSAGE
        ========================== */}

        {success ? (

          <div className="text-center py-6">

            <div className="text-4xl mb-3">
              ✓
            </div>

            <p className="text-emerald-600 font-medium">
              Registration submitted successfully.
            </p>

            <p className="text-sm text-slate-500 mt-2 leading-6">
              Your doctor account is now waiting for administrator approval.
              You will be able to log in after approval.
            </p>

            <p className="text-sm text-slate-400 mt-2">
              Redirecting to login…
            </p>

          </div>

        ) : (

          <form
            onSubmit={handleSubmit}
            className="space-y-4"
          >

            {/* =========================
                FULL NAME
            ========================== */}

            <div>

              <label className="text-sm font-medium block mb-1">
                Full name
              </label>

              <input
                type="text"
                required
                className="input-field"
                value={form.full_name}
                onChange={update('full_name')}
                placeholder="Enter your full name"
              />

            </div>


            {/* =========================
                EMAIL
            ========================== */}

            <div>

              <label className="text-sm font-medium block mb-1">
                Email
              </label>

              <input
                type="email"
                required
                className="input-field"
                value={form.email}
                onChange={update('email')}
                placeholder="Enter your email"
              />

            </div>


            {/* =========================
                DOCTOR ID
            ========================== */}

            <div>

              <label className="text-sm font-medium block mb-1">
                Doctor ID
              </label>

              <input
                type="text"
                required
                className="input-field"
                value={form.doctor_id}
                onChange={update('doctor_id')}
                placeholder="Enter your Doctor ID"
              />

            </div>


            {/* =========================
                VERIFICATION CODE
            ========================== */}

            <div>

              <label className="text-sm font-medium block mb-1">
                Verification code
              </label>

              <input
                type="text"
                required
                className="input-field"
                value={form.verification_code}
                onChange={update('verification_code')}
                placeholder="Enter verification code"
              />

              <p className="text-xs text-slate-400 mt-1">
                Your Doctor ID and verification code will be checked before approval.
              </p>

            </div>


            {/* =========================
                PASSWORD
            ========================== */}

            <div>

              <label className="text-sm font-medium block mb-1">
                Password
              </label>

              <input
                type="password"
                required
                className="input-field"
                value={form.password}
                onChange={update('password')}
                placeholder="Create a password"
              />

              <p className="text-xs text-slate-400 mt-1">
                At least 8 characters, with uppercase, lowercase, a digit,
                and a special character.
              </p>

            </div>


            {/* =========================
                CONFIRM PASSWORD
            ========================== */}

            <div>

              <label className="text-sm font-medium block mb-1">
                Confirm password
              </label>

              <input
                type="password"
                required
                className="input-field"
                value={form.confirm_password}
                onChange={update('confirm_password')}
                placeholder="Confirm your password"
              />

            </div>


            {/* =========================
                APPROVAL NOTICE
            ========================== */}

            <div className="rounded-lg bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900 px-4 py-3">

              <p className="text-xs text-blue-700 dark:text-blue-300 leading-5">
                Your account will be reviewed by an administrator before
                you can access the NeuroScan AI system.
              </p>

            </div>


            {/* =========================
                ERROR
            ========================== */}

            <ErrorAlert message={error} />


            {/* =========================
                SUBMIT
            ========================== */}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading
                ? 'Submitting registration…'
                : 'Submit Registration'}
            </button>

          </form>
        )}


        {/* =========================
            LOGIN LINK
        ========================== */}

        <p className="text-sm text-center mt-6 text-slate-500 dark:text-slate-400">

          Already have an account?{' '}

          <Link
            to="/login"
            className="text-brand-600 font-medium"
          >
            Log in
          </Link>

        </p>

      </div>

    </div>
  )
}