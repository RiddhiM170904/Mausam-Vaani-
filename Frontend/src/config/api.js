// API configuration and base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const API_ENDPOINTS = {
  // Health check
  health: `${API_BASE_URL}/health`,
  
  // Main prediction endpoint
  predict: `${API_BASE_URL}/predict`,
  
  // Root info
  root: `${API_BASE_URL}/`,
}

export default API_BASE_URL
