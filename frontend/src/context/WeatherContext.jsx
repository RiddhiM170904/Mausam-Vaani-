import { createContext, useContext, useState, useEffect } from 'react'

const WeatherContext = createContext(null)

export const useWeather = () => {
  const context = useContext(WeatherContext)
  if (!context) {
    throw new Error('useWeather must be used within a WeatherProvider')
  }
  return context
}

export const WeatherProvider = ({ children }) => {
  const [currentWeather, setCurrentWeather] = useState(null)
  const [hourlyForecast, setHourlyForecast] = useState([])
  const [dailyForecast, setDailyForecast] = useState([])
  const [insights, setInsights] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  const [location, setLocation] = useState({ lat: null, lon: null, name: 'Kothri Kalan, MP' })

  // Mock data for development
  const mockWeatherData = {
    current: {
      condition: 'sunny',
      temperature: 32,
      realFeel: 35,
      humidity: 65,
      windSpeed: 12,
      aqi: 85,
      location: 'Kothri Kalan, MP',
      description: 'Partly Cloudy',
      high: 34,
      low: 22,
    },
    hourly: [
      { time: 'Now', temp: 32, condition: 'sunny', rainChance: 0 },
      { time: '1 PM', temp: 33, condition: 'sunny', rainChance: 5 },
      { time: '2 PM', temp: 34, condition: 'cloudy', rainChance: 10 },
      { time: '3 PM', temp: 33, condition: 'cloudy', rainChance: 20 },
      { time: '4 PM', temp: 31, condition: 'cloudy', rainChance: 35 },
      { time: '5 PM', temp: 29, condition: 'rain', rainChance: 60 },
      { time: '6 PM', temp: 27, condition: 'rain', rainChance: 75 },
      { time: '7 PM', temp: 26, condition: 'rain', rainChance: 50 },
    ],
    daily: [
      { day: 'Today', date: 'Jan 26', high: 32, low: 22, condition: 'sunny', rainChance: 10 },
      { day: 'Mon', date: 'Jan 27', high: 30, low: 21, condition: 'cloudy', rainChance: 25 },
      { day: 'Tue', date: 'Jan 28', high: 28, low: 20, condition: 'rain', rainChance: 80 },
      { day: 'Wed', date: 'Jan 29', high: 26, low: 19, condition: 'rain', rainChance: 90 },
      { day: 'Thu', date: 'Jan 30', high: 27, low: 18, condition: 'cloudy', rainChance: 40 },
      { day: 'Fri', date: 'Jan 31', high: 29, low: 19, condition: 'sunny', rainChance: 10 },
      { day: 'Sat', date: 'Feb 1', high: 31, low: 20, condition: 'sunny', rainChance: 5 },
    ],
  }

  // Fetch weather data
  const fetchWeather = async (lat, lon) => {
    setIsLoading(true)
    setError(null)
    
    try {
      // In production, replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      setCurrentWeather(mockWeatherData.current)
      setHourlyForecast(mockWeatherData.hourly)
      setDailyForecast(mockWeatherData.daily)
    } catch (err) {
      setError('Failed to fetch weather data')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch AI insights based on persona
  const fetchInsights = async (persona) => {
    try {
      // In production, call Gemini API
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const personaInsights = {
        farmer: [
          {
            type: 'farmer',
            severity: 'warning',
            title: '🌾 Crop Advisory',
            message: 'Humidity is high (85%). Risk of fungal blight on wheat crop. Avoid irrigation today.',
            timestamp: 'Updated 10 min ago',
          },
          {
            type: 'farmer',
            severity: 'info',
            title: '💧 Water Management',
            message: 'Rain expected tomorrow. Save irrigation water for next week.',
            timestamp: 'Updated 30 min ago',
          },
        ],
        commuter: [
          {
            type: 'commuter',
            severity: 'warning',
            title: '🛵 Travel Alert',
            message: 'Heavy rain expected at 6 PM. Leave office by 5:30 PM to avoid traffic.',
            timestamp: 'Updated 15 min ago',
          },
        ],
        general: [
          {
            type: 'general',
            severity: 'info',
            title: '☂️ Daily Tip',
            message: 'Carry an umbrella. 60% chance of evening showers.',
            timestamp: 'Updated 20 min ago',
          },
        ],
      }
      
      setInsights(personaInsights[persona] || personaInsights.general)
    } catch (err) {
      console.error('Failed to fetch insights:', err)
    }
  }

  // Auto-detect location
  const detectLocation = () => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error('Geolocation not supported'))
        return
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords
          setLocation({ lat: latitude, lon: longitude, name: 'Detected Location' })
          resolve({ lat: latitude, lon: longitude })
        },
        (error) => {
          reject(error)
        }
      )
    })
  }

  // Initial fetch
  useEffect(() => {
    fetchWeather()
  }, [])

  const value = {
    currentWeather,
    hourlyForecast,
    dailyForecast,
    insights,
    isLoading,
    error,
    location,
    fetchWeather,
    fetchInsights,
    detectLocation,
    setLocation,
  }

  return (
    <WeatherContext.Provider value={value}>
      {children}
    </WeatherContext.Provider>
  )
}

export default WeatherContext
