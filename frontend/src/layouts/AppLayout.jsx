import React, { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth.jsx'


/* =====================================================
   USER NAVIGATION
===================================================== */

const userNav = [
  {
    to: '/dashboard',
    label: 'Dashboard',
    icon: '📊',
  },
  {
    to: '/analyze',
    label: 'New Analysis',
    icon: '🧠',
  },
  {
    to: '/history',
    label: 'Analysis History',
    icon: '🗂️',
  },
  {
    to: '/profile',
    label: 'Profile',
    icon: '👤',
  },
]


/* =====================================================
   ADMIN NAVIGATION
===================================================== */

const adminNav = [
  {
    to: '/admin',
    label: 'Admin Dashboard',
    icon: '🛡️',
  },
  {
    to: '/admin/users',
    label: 'Users',
    icon: '👥',
  },
  {
    to: '/admin/logs',
    label: 'System Logs',
    icon: '📜',
  },
]


/* =====================================================
   APP LAYOUT
===================================================== */

export default function AppLayout({ children }) {

  const { user, logout } = useAuth()

  const location = useLocation()

  const navigate = useNavigate()

  const [sidebarOpen, setSidebarOpen] = useState(false)


  /* ===================================================
     LOGOUT
  =================================================== */

  const handleLogout = async () => {

    await logout()

    navigate('/login')
  }


  /* ===================================================
     NAVIGATION ITEMS
  =================================================== */

  const navItems =
    user?.role === 'admin'
      ? [...userNav, ...adminNav]
      : userNav


  return (

    <div className="
      min-h-screen
      flex
      bg-slate-950
      text-slate-100
    ">


      {/* =================================================
          SIDEBAR
      ================================================== */}

      <aside
        className={`
          fixed
          lg:static
          z-30
          inset-y-0
          left-0
          w-64

          bg-slate-950

          border-r
          border-slate-800

          transform

          ${sidebarOpen
            ? 'translate-x-0'
            : '-translate-x-full'
          }

          lg:translate-x-0

          transition-transform
          duration-200
        `}
      >


        {/* ===============================================
            LOGO
        ================================================ */}

        <div className="
          h-16
          px-5
          flex
          items-center
          gap-3
          border-b
          border-slate-800
        ">

          <span className="text-2xl">
            🧠
          </span>

          <span className="
            font-bold
            text-lg
            text-white
          ">
            NeuroScan AI
          </span>

        </div>


        {/* ===============================================
            NAVIGATION
        ================================================ */}

        <nav className="
          p-3
          space-y-1
        ">

          {navItems.map((item) => (

            <Link
              key={item.to}
              to={item.to}
              onClick={() => setSidebarOpen(false)}

              className={`
                flex
                items-center
                gap-3
                px-3
                py-3
                rounded-lg
                text-sm
                font-medium
                transition

                ${
                  location.pathname === item.to

                    ? `
                      bg-blue-600/20
                      text-blue-400
                      border
                      border-blue-500/20
                    `

                    : `
                      text-slate-400
                      hover:bg-slate-800
                      hover:text-white
                    `
                }
              `}
            >

              <span className="text-lg">
                {item.icon}
              </span>

              <span>
                {item.label}
              </span>

            </Link>

          ))}

        </nav>


        {/* ===============================================
            LOGOUT
        ================================================ */}

        <div className="
          absolute
          bottom-0
          left-0
          right-0
          p-3
          border-t
          border-slate-800
        ">

          <button
            onClick={handleLogout}

            className="
              w-full
              flex
              items-center
              gap-3
              px-3
              py-3
              rounded-lg

              text-sm
              font-medium
              text-slate-400

              hover:bg-red-500/10
              hover:text-red-400

              transition
            "
          >

            <span className="text-lg">
              🚪
            </span>

            <span>
              Logout
            </span>

          </button>

        </div>

      </aside>


      {/* =================================================
          MAIN AREA
      ================================================== */}

      <div className="
        flex-1
        flex
        flex-col
        min-w-0
      ">


        {/* ===============================================
            HEADER
        ================================================ */}

        <header className="
          h-16

          border-b
          border-slate-800

          bg-slate-950

          flex
          items-center
          justify-between

          px-4
          lg:px-6
        ">


          {/* MOBILE MENU */}

          <button
            className="
              lg:hidden
              text-xl
              text-slate-300
              hover:text-white
            "

            onClick={() =>
              setSidebarOpen((s) => !s)
            }
          >
            ☰
          </button>


          <div className="flex-1" />


          {/* =============================================
              USER INFORMATION
          ============================================== */}

          <div className="
            flex
            items-center
            gap-3
          ">


            {/* USER ICON */}

            <div className="
              w-9
              h-9
              rounded-full

              bg-blue-600

              flex
              items-center
              justify-center

              text-white
              font-semibold
            ">
              {user?.full_name?.charAt(0)?.toUpperCase() || 'U'}
            </div>


            {/* USER DETAILS */}

            <div className="
              text-sm
              text-right
              hidden
              sm:block
            ">

              <div className="
                font-medium
                text-white
              ">
                {user?.full_name || 'User'}
              </div>

              <div className="
                text-slate-500
                text-xs
              ">
                {user?.role === 'admin'
                  ? 'Administrator'
                  : 'Researcher'}
              </div>

            </div>

          </div>

        </header>


        {/* ===============================================
            PAGE CONTENT
        ================================================ */}

        <main className="
          flex-1
          p-4
          lg:p-6
          overflow-y-auto
          bg-slate-950
        ">

          {children}

        </main>

      </div>

    </div>
  )
}