import React, { useState } from 'react'
import { useAuth } from '../hooks/useAuth.jsx'
import api from '../services/api'
import { ErrorAlert } from '../components/Common.jsx'

export default function Profile() {
  const { user, logout } = useAuth()

  const [current_password, setCurrent] = useState('')
  const [new_password, setNewPass] = useState('')
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // =========================================================
  // CHANGE PASSWORD
  // =========================================================

  const handleChangePassword = async (e) => {
    e.preventDefault()

    setMsg('')
    setError('')
    setLoading(true)

    try {
      await api.post(
        '/api/auth/change-password',
        {
          current_password,
          new_password,
        }
      )

      setMsg(
        'Password updated successfully.'
      )

      setCurrent('')
      setNewPass('')

    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Failed to update password.'
      )

    } finally {
      setLoading(false)
    }
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
        max-w-3xl
        mx-auto
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
            Profile
          </h1>

          <p className="
            text-slate-400
            text-sm
            sm:text-base
            mt-2
          ">
            Manage your account details.
          </p>

        </div>

        {/* =====================================================
            ACCOUNT DETAILS
        ====================================================== */}

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

          <div className="
            flex
            items-center
            gap-4
            mb-6
          ">

            <div className="
              w-12
              h-12
              rounded-xl
              bg-blue-500/10
              border
              border-blue-500/20
              flex
              items-center
              justify-center
              text-2xl
            ">
              👤
            </div>

            <div>

              <h2 className="
                text-xl
                font-semibold
                text-white
              ">
                Account Information
              </h2>

              <p className="
                text-sm
                text-slate-400
                mt-1
              ">
                Your NeuroScan account details.
              </p>

            </div>

          </div>

          <div>

            <Row
              label="Full name"
              value={user?.full_name}
            />

            <Row
              label="Email"
              value={user?.email}
            />

            <Row
              label="Role"
              value={
                user?.role === 'admin'
                  ? 'Administrator'
                  : 'Doctor'
              }
            />

            <Row
              label="Account created"
              value={
                user?.created_at
                  ? new Date(
                      user.created_at
                    ).toLocaleDateString()
                  : '—'
              }
              last
            />

          </div>

        </div>

        {/* =====================================================
            CHANGE PASSWORD
        ====================================================== */}

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

          <div className="
            flex
            items-center
            gap-4
            mb-6
          ">

            <div className="
              w-12
              h-12
              rounded-xl
              bg-blue-500/10
              border
              border-blue-500/20
              flex
              items-center
              justify-center
              text-2xl
            ">
              🔐
            </div>

            <div>

              <h2 className="
                text-xl
                font-semibold
                text-white
              ">
                Change Password
              </h2>

              <p className="
                text-sm
                text-slate-400
                mt-1
              ">
                Update your account password securely.
              </p>

            </div>

          </div>

          <form
            onSubmit={handleChangePassword}
            className="space-y-5"
          >

            {/* CURRENT PASSWORD */}

            <div>

              <label className="
                block
                text-sm
                font-medium
                text-slate-300
                mb-2
              ">
                Current password
              </label>

              <input
                type="password"
                required
                value={current_password}
                onChange={(e) =>
                  setCurrent(e.target.value)
                }
                placeholder="Enter current password"
                className="
                  w-full
                  bg-[#081b36]
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

            {/* NEW PASSWORD */}

            <div>

              <label className="
                block
                text-sm
                font-medium
                text-slate-300
                mb-2
              ">
                New password
              </label>

              <input
                type="password"
                required
                minLength={8}
                value={new_password}
                onChange={(e) =>
                  setNewPass(e.target.value)
                }
                placeholder="Enter new password"
                className="
                  w-full
                  bg-[#081b36]
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

              <p className="
                text-xs
                text-slate-500
                mt-2
              ">
                Minimum 8 characters.
              </p>

            </div>

            {/* SUCCESS MESSAGE */}

            {msg && (

              <div className="
                rounded-xl
                border
                border-emerald-500/20
                bg-emerald-500/10
                px-4
                py-3
              ">

                <p className="
                  text-sm
                  text-emerald-400
                  font-medium
                ">
                  {msg}
                </p>

              </div>

            )}

            {/* ERROR */}

            <ErrorAlert message={error} />

            {/* UPDATE BUTTON */}

            <button
              type="submit"
              disabled={loading}
              className={`
                w-full
                sm:w-auto
                px-6
                py-3
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
                ? 'Updating…'
                : 'Update Password'}
            </button>

          </form>

        </div>

        {/* =====================================================
            LOGOUT
        ====================================================== */}

        <button
          type="button"
          onClick={logout}
          className="
            w-full
            rounded-xl
            border
            border-red-500/20
            bg-red-500/10
            text-red-400
            hover:bg-red-500/20
            hover:text-red-300
            font-semibold
            px-5
            py-3
            transition
          "
        >
          Logout
        </button>

      </main>

    </div>
  )
}


/* =====================================================
   ACCOUNT INFORMATION ROW
===================================================== */

function Row({
  label,
  value,
  last = false,
}) {
  return (
    <div
      className={`
        flex
        flex-col
        sm:flex-row
        sm:items-center
        sm:justify-between
        gap-2
        py-4
        ${
          !last
            ? 'border-b border-blue-900/40'
            : ''
        }
      `}
    >

      <span className="
        text-sm
        text-slate-400
      ">
        {label}
      </span>

      <span className="
        text-sm
        font-medium
        text-white
        break-all
        sm:text-right
      ">
        {value || '—'}
      </span>

    </div>
  )
}