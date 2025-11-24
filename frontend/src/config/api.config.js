// API Configuration for Mausam-Vaani Frontend

const API_CONFIG = {
    // Backend API base URL
    BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:5000',

    // API endpoints
    ENDPOINTS: {
        HEALTH: '/health',
        PREDICT_WEATHER: '/api/predict-weather',
        GET_INSIGHT: '/api/get-insight',
        PING: '/api/ping'
    },

    // Request timeout (ms)
    TIMEOUT: 30000,

    // Retry configuration
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000
};

export default API_CONFIG;
