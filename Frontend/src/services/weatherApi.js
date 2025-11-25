import { API_ENDPOINTS } from '../config/api'

/**
 * Fetch weather prediction and personalized insights
 * @param {Object} params - Prediction parameters
 * @param {Object} params.weatherInput - Weather input data (latitude, longitude, location_name)
 * @param {Object} params.userContext - User context (profession, additional_context)
 * @param {number} params.forecastHours - Number of hours to forecast (default: 24)
 * @returns {Promise<Object>} - Prediction response
 */
export const getWeatherPrediction = async ({
  weatherInput,
  userContext,
  forecastHours = 24
}) => {
  try {
    const response = await fetch(API_ENDPOINTS.predict, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        weather_input: weatherInput,
        user_context: userContext,
        forecast_hours: forecastHours,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to fetch prediction')
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Weather prediction error:', error)
    throw error
  }
}

/**
 * Check API health status
 * @returns {Promise<Object>} - Health status
 */
export const checkHealth = async () => {
  try {
    const response = await fetch(API_ENDPOINTS.health)
    
    if (!response.ok) {
      throw new Error('Health check failed')
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Health check error:', error)
    throw error
  }
}

/**
 * Get API information
 * @returns {Promise<Object>} - API info
 */
export const getApiInfo = async () => {
  try {
    const response = await fetch(API_ENDPOINTS.root)
    
    if (!response.ok) {
      throw new Error('Failed to fetch API info')
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('API info error:', error)
    throw error
  }
}

// Example usage functions for different professions
export const getPredictionForFarmer = (latitude, longitude, locationName, crop) => {
  return getWeatherPrediction({
    weatherInput: {
      latitude,
      longitude,
      location_name: locationName,
    },
    userContext: {
      profession: 'Farmer',
      additional_context: {
        crop,
      },
    },
    forecastHours: 24,
  })
}

export const getPredictionForCommuter = (latitude, longitude, locationName) => {
  return getWeatherPrediction({
    weatherInput: {
      latitude,
      longitude,
      location_name: locationName,
    },
    userContext: {
      profession: 'Commuter',
    },
    forecastHours: 24,
  })
}

export const getPredictionForConstructionWorker = (latitude, longitude, locationName) => {
  return getWeatherPrediction({
    weatherInput: {
      latitude,
      longitude,
      location_name: locationName,
    },
    userContext: {
      profession: 'Construction Worker',
    },
    forecastHours: 24,
  })
}

export const getGeneralPrediction = (latitude, longitude, locationName) => {
  return getWeatherPrediction({
    weatherInput: {
      latitude,
      longitude,
      location_name: locationName,
    },
    userContext: {
      profession: 'General',
    },
    forecastHours: 24,
  })
}
