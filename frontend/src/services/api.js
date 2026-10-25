import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({ baseURL: API_BASE_URL })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('access_token')
      localStorage.removeItem('user')
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(err)
  }
)

export default api

export function authImageUrl(path) {
  // Helper for constructing image URLs; actual fetch done via api client with blob.
  return `${API_BASE_URL}${path}`
}

export async function fetchImageBlobUrl(path) {
  const res = await api.get(path, { responseType: 'blob' })
  return URL.createObjectURL(res.data)
}
