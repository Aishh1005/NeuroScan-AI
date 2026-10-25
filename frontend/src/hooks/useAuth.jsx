import React, {
  createContext,
  useContext,
  useState,
  useCallback,
} from 'react'

import api from '../services/api'


// =========================================================
// AUTH CONTEXT
// =========================================================

const AuthContext = createContext(null)


export function AuthProvider({ children }) {

  // -------------------------------------------------------
  // Load previously logged-in user
  // -------------------------------------------------------

  const [user, setUser] = useState(() => {

    const stored = localStorage.getItem('user')

    return stored ? JSON.parse(stored) : null
  })


  // =======================================================
  // LOGIN
  // =======================================================

  const login = useCallback(
    async (email, password) => {

      const res = await api.post(
        '/api/auth/login',
        {
          email,
          password,
        }
      )

      // Save token
      localStorage.setItem(
        'access_token',
        res.data.access_token
      )

      // Save user
      localStorage.setItem(
        'user',
        JSON.stringify(res.data.user)
      )

      // Update React state
      setUser(res.data.user)

      return res.data.user
    },
    []
  )


  // =======================================================
  // DOCTOR REGISTRATION
  // =======================================================

  const register = useCallback(
    async (
      full_name,
      email,
      password,
      confirm_password,
      doctor_id,
      verification_code
    ) => {

      const res = await api.post(
        '/api/auth/register',
        {
          full_name,
          email,
          password,
          confirm_password,
          doctor_id,
          verification_code,
        }
      )

      return res.data
    },
    []
  )


  // =======================================================
  // LOGOUT
  // =======================================================

  const logout = useCallback(
    async () => {

      try {

        await api.post(
          '/api/auth/logout'
        )

      } catch (e) {

        // Ignore network errors during logout
      }

      // Remove stored authentication data
      localStorage.removeItem('access_token')
      localStorage.removeItem('user')

      // Clear React user state
      setUser(null)
    },
    []
  )


  // =======================================================
  // PROVIDER
  // =======================================================

  return (

    <AuthContext.Provider
      value={{
        user,
        setUser,
        login,
        register,
        logout,
      }}
    >

      {children}

    </AuthContext.Provider>
  )
}


// =========================================================
// USE AUTH HOOK
// =========================================================

export function useAuth() {

  return useContext(AuthContext)
}