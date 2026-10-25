import React from 'react'
import { Routes, Route } from 'react-router-dom'

import Landing from './pages/Landing.jsx'
import Login from './pages/Login.jsx'
import Signup from './pages/Signup.jsx'

import Dashboard from './pages/Dashboard.jsx'
import Analyze from './pages/Analyze.jsx'
import Result from './pages/Result.jsx'
import History from './pages/History.jsx'
import Profile from './pages/Profile.jsx'

import AdminDashboard from './pages/AdminDashboard.jsx'
import AdminUsers from './pages/AdminUsers.jsx'
import AdminLogs from './pages/AdminLogs.jsx'

import NotFound from './pages/NotFound.jsx'

import ProtectedRoute from './components/ProtectedRoute.jsx'
import AppLayout from './layouts/AppLayout.jsx'


function Protected({ children, adminOnly }) {
  return (
    <ProtectedRoute adminOnly={adminOnly}>
      <AppLayout>{children}</AppLayout>
    </ProtectedRoute>
  )
}


export default function App() {
  return (
    <Routes>

      {/* =========================
          PUBLIC PAGES
      ========================== */}

      <Route
        path="/"
        element={<Landing />}
      />

      <Route
        path="/login"
        element={<Login />}
      />

      <Route
        path="/signup"
        element={<Signup />}
      />


      {/* =========================
          USER PAGES
      ========================== */}

      <Route
        path="/dashboard"
        element={
          <Protected>
            <Dashboard />
          </Protected>
        }
      />

      <Route
        path="/analyze"
        element={
          <Protected>
            <Analyze />
          </Protected>
        }
      />

      <Route
        path="/result/:id"
        element={
          <Protected>
            <Result />
          </Protected>
        }
      />

      <Route
        path="/history"
        element={
          <Protected>
            <History />
          </Protected>
        }
      />

      <Route
        path="/profile"
        element={
          <Protected>
            <Profile />
          </Protected>
        }
      />


      {/* =========================
          ADMIN PAGES
      ========================== */}

      <Route
        path="/admin"
        element={
          <Protected adminOnly>
            <AdminDashboard />
          </Protected>
        }
      />

      <Route
        path="/admin/users"
        element={
          <Protected adminOnly>
            <AdminUsers />
          </Protected>
        }
      />

      <Route
        path="/admin/logs"
        element={
          <Protected adminOnly>
            <AdminLogs />
          </Protected>
        }
      />


      {/* =========================
          404
      ========================== */}

      <Route
        path="*"
        element={<NotFound />}
      />

    </Routes>
  )
}