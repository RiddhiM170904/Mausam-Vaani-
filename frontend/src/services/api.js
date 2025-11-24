/**
 * API Service for Mausam-Vaani Frontend
 * Handles all communication with the Flask AI Backend
 */

import axios from 'axios';
import API_CONFIG from '../config/api.config';

// Create axios instance with base configuration
const apiClient = axios.create({
    baseURL: API_CONFIG.BASE_URL,
    timeout: API_CONFIG.TIMEOUT,
    headers: {
        'Content-Type': 'application/json'
    }
});

// Request interceptor for logging
apiClient.interceptors.request.use(
    (config) => {
        console.log(`API Request: ${config.method.toUpperCase()} ${config.url}`);
        return config;
    },
    (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
    }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
    (response) => {
        console.log(`API Response: ${response.config.url}`, response.data);
        return response;
    },
    (error) => {
        console.error('API Response Error:', error);

        if (error.response) {
            // Server responded with error status
            const { status, data } = error.response;
            console.error(`Server Error ${status}:`, data);
        } else if (error.request) {
            // Request made but no response
            console.error('No response from server. Is the backend running?');
        } else {
            // Request setup error
            console.error('Request error:', error.message);
        }

        return Promise.reject(error);
    }
);

/**
 * API Service Object
 */
const apiService = {
    /**
     * Check backend health status
     * @returns {Promise<Object>} Health status
     */
    checkHealth: async () => {
        try {
            const response = await apiClient.get(API_CONFIG.ENDPOINTS.HEALTH);
            return {
                success: true,
                data: response.data
            };
        } catch (error) {
            return {
                success: false,
                error: error.message || 'Failed to connect to backend'
            };
        }
    },

    /**
     * Predict weather based on historical data
     * @param {Object} historicalData - Historical weather data (168 hours)
     * @param {number} forecastSteps - Number of hours to predict (default: 24)
     * @returns {Promise<Object>} Weather predictions
     */
    predictWeather: async (historicalData, forecastSteps = 24) => {
        try {
            const response = await apiClient.post(API_CONFIG.ENDPOINTS.PREDICT_WEATHER, {
                historical_data: historicalData,
                forecast_steps: forecastSteps
            });

            if (response.data.success) {
                return {
                    success: true,
                    data: response.data
                };
            } else {
                return {
                    success: false,
                    error: response.data.error || 'Failed to predict weather'
                };
            }
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.error || error.message || 'Failed to predict weather'
            };
        }
    },

    /**
     * Get personalized weather insight
     * @param {Object} params - Request parameters
     * @param {number} params.latitude - Latitude
     * @param {number} params.longitude - Longitude
     * @param {string} params.city - City name
     * @param {string} params.userProfession - User profession
     * @param {Object} params.userContext - Additional user context
     * @param {Object} params.historicalData - Historical weather data
     * @param {number} params.forecastSteps - Forecast steps
     * @returns {Promise<Object>} Personalized insight
     */
    getInsight: async (params) => {
        try {
            const requestBody = {
                latitude: params.latitude,
                longitude: params.longitude,
                city: params.city,
                user_profession: params.userProfession,
                user_context: params.userContext || {},
                historical_data: params.historicalData,
                forecast_steps: params.forecastSteps || 24
            };

            const response = await apiClient.post(API_CONFIG.ENDPOINTS.GET_INSIGHT, requestBody);

            if (response.data.success) {
                return {
                    success: true,
                    data: response.data
                };
            } else {
                return {
                    success: false,
                    error: response.data.error || 'Failed to get insight'
                };
            }
        } catch (error) {
            return {
                success: false,
                error: error.response?.data?.error || error.message || 'Failed to get insight'
            };
        }
    },

    /**
     * Ping API to check if it's alive
     * @returns {Promise<Object>} Ping response
     */
    ping: async () => {
        try {
            const response = await apiClient.get(API_CONFIG.ENDPOINTS.PING);
            return {
                success: true,
                data: response.data
            };
        } catch (error) {
            return {
                success: false,
                error: error.message || 'Failed to ping API'
            };
        }
    }
};

export default apiService;
