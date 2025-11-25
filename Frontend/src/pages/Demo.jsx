import { useState, useEffect } from 'react'
import { MapPin, Cloud, Droplets, Wind, AlertCircle, Loader } from 'lucide-react'
import { getWeatherPrediction, checkHealth } from '../services/weatherApi'

const Demo = () => {
  const [loading, setLoading] = useState(false)
  const [detectingLocation, setDetectingLocation] = useState(false)
  const [apiStatus, setApiStatus] = useState(null)
  const [formData, setFormData] = useState({
    locationName: '',
    district: '',
    village: '',
    latitude: null,
    longitude: null,
    occupation: 'General',
    plannedActivity: '',
    activityTime: 'morning',
    duration: '2-4 hours',
    concerns: '',
    forecastHours: 24,
  })
  const [locationDetails, setLocationDetails] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  // Check API health on mount
  useEffect(() => {
    checkApiHealth()
  }, [])

  const detectMyLocation = async () => {
    setDetectingLocation(true)
    setError(null)

    try {
      // Get browser geolocation
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 10000,
        })
      })

      const lat = position.coords.latitude
      const lon = position.coords.longitude

      // Reverse geocode to get location details
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=18&addressdetails=1`
      )
      
      if (!response.ok) throw new Error('Failed to get location details')
      
      const data = await response.json()
      const address = data.address || {}

      // Extract village, district, and city info
      const village = address.village || address.hamlet || address.suburb || ''
      const district = address.county || address.state_district || ''
      const city = address.city || address.town || address.municipality || district
      
      const locationName = village 
        ? `${village}, ${district || city}` 
        : city

      setFormData(prev => ({
        ...prev,
        locationName: locationName,
        district: district,
        village: village,
        latitude: lat,
        longitude: lon,
      }))

      setLocationDetails({
        fullAddress: data.display_name,
        village: village,
        district: district,
        city: city,
        state: address.state || '',
        country: address.country || '',
        coordinates: { lat, lon },
      })

    } catch (err) {
      console.error('Location detection error:', err)
      setError(
        err.code === 1 
          ? 'Location permission denied. Please enter location manually.' 
          : 'Could not detect location. Please enter manually.'
      )
    } finally {
      setDetectingLocation(false)
    }
  }

  const checkApiHealth = async () => {
    try {
      const health = await checkHealth()
      setApiStatus(health)
    } catch (err) {
      setApiStatus({ status: 'offline', error: err.message })
    }
  }

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      // Build comprehensive user context for LLM
      const additionalContext = {
        planned_activity: formData.plannedActivity,
        activity_time: formData.activityTime,
        duration: formData.duration,
        specific_concerns: formData.concerns,
      }

      // Add location context
      if (locationDetails) {
        additionalContext.location_type = locationDetails.village ? 'Village' : 'City'
        additionalContext.village = locationDetails.village
        additionalContext.district = locationDetails.district
        additionalContext.state = locationDetails.state
      }

      const result = await getWeatherPrediction({
        weatherInput: {
          location_name: formData.locationName,
          latitude: formData.latitude,
          longitude: formData.longitude,
        },
        userContext: {
          profession: formData.occupation,
          additional_context: additionalContext,
        },
        forecastHours: parseInt(formData.forecastHours),
      })

      setPrediction(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const occupations = [
    'Farmer/Agriculture',
    'Daily Commuter/Office Worker', 
    'Construction/Outdoor Worker',
    'Sports/Fitness Enthusiast',
    'Student',
    'Delivery/Logistics',
    'Event Planner',
    'Photographer/Videographer',
    'Tourist/Traveler',
    'General/Other'
  ]

  const activityTimes = [
    { value: 'morning', label: 'Morning (6 AM - 12 PM)' },
    { value: 'afternoon', label: 'Afternoon (12 PM - 5 PM)' },
    { value: 'evening', label: 'Evening (5 PM - 9 PM)' },
    { value: 'night', label: 'Night (9 PM - 6 AM)' },
    { value: 'all_day', label: 'All Day' },
  ]

  const durations = ['< 1 hour', '1-2 hours', '2-4 hours', '4-8 hours', 'Full day', 'Multiple days']

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Live Weather Prediction Demo
          </h1>
          <p className="text-lg text-gray-600">
            Experience hyperlocal weather intelligence with AI-powered personalized insights
          </p>
          
          {/* API Status */}
          <div className="mt-4 inline-flex items-center px-4 py-2 rounded-lg bg-white shadow-md">
            <div className={`w-3 h-3 rounded-full mr-2 ${apiStatus?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm font-medium">
              {apiStatus?.status === 'healthy' ? 'API Connected' : 'API Offline'}
            </span>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Plan Your Activity</h2>
            
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Auto-Detect Location */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  📍 Location (Hyperlocal)
                </label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={detectMyLocation}
                    disabled={detectingLocation}
                    className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors font-medium flex items-center space-x-2"
                  >
                    {detectingLocation ? (
                      <>
                        <Loader className="h-4 w-4 animate-spin" />
                        <span>Detecting...</span>
                      </>
                    ) : (
                      <>
                        <MapPin className="h-4 w-4" />
                        <span>Detect My Location</span>
                      </>
                    )}
                  </button>
                  <div className="text-xs text-gray-500 flex items-center">
                    Or enter manually below
                  </div>
                </div>
              </div>

              {/* Location Details Display */}
              {locationDetails && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm">
                  <div className="font-semibold text-blue-900 mb-2 flex items-center space-x-2">
                    <MapPin className="h-4 w-4" />
                    <span>Detected Location (Village-Level Precision)</span>
                  </div>
                  <div className="space-y-1 text-blue-800">
                    {locationDetails.village && (
                      <p>🏘️ Village: <span className="font-medium">{locationDetails.village}</span></p>
                    )}
                    {locationDetails.district && (
                      <p>🏛️ District: <span className="font-medium">{locationDetails.district}</span></p>
                    )}
                    {locationDetails.city && locationDetails.city !== locationDetails.district && (
                      <p>🏙️ City: <span className="font-medium">{locationDetails.city}</span></p>
                    )}
                    {locationDetails.state && (
                      <p>📍 State: <span className="font-medium">{locationDetails.state}</span></p>
                    )}
                    <p className="text-xs text-blue-600 mt-2">
                      Coordinates: {locationDetails.coordinates.lat.toFixed(4)}, {locationDetails.coordinates.lon.toFixed(4)}
                    </p>
                  </div>
                </div>
              )}

              {/* Manual Location Input */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Or Enter Location Manually
                </label>
                <input
                  type="text"
                  name="locationName"
                  value={formData.locationName}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                  placeholder="e.g., Sehore, MP or specific village name"
                  required
                />
                <p className="mt-1 text-xs text-gray-500">
                  City, district, or village name (supports hyperlocal)
                </p>
              </div>

              {/* Occupation */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  👤 Your Occupation/Role
                </label>
                <select
                  name="occupation"
                  value={formData.occupation}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                >
                  {occupations.map(occ => (
                    <option key={occ} value={occ}>{occ}</option>
                  ))}
                </select>
              </div>

              {/* Planned Activity */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  🎯 What are you planning?
                </label>
                <input
                  type="text"
                  name="plannedActivity"
                  value={formData.plannedActivity}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                  placeholder="e.g., Wedding, Farming, Travel, Sports, Delivery"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Optional: Describe your planned activity for personalized advice
                </p>
              </div>

              {/* Activity Time */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  ⏰ When?
                </label>
                <select
                  name="activityTime"
                  value={formData.activityTime}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                >
                  {activityTimes.map(time => (
                    <option key={time.value} value={time.value}>{time.label}</option>
                  ))}
                </select>
              </div>

              {/* Duration */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  ⌛ How long?
                </label>
                <select
                  name="duration"
                  value={formData.duration}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                >
                  {durations.map(dur => (
                    <option key={dur} value={dur}>{dur}</option>
                  ))}
                </select>
              </div>

              {/* Specific Concerns */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  💭 Specific Concerns?
                </label>
                <textarea
                  name="concerns"
                  value={formData.concerns}
                  onChange={handleInputChange}
                  rows="2"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none resize-none"
                  placeholder="e.g., Worried about rain, heat, wind, etc."
                />
              </div>

              {/* Forecast Hours */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  📊 Forecast Duration: {formData.forecastHours} hours
                </label>
                <input
                  type="range"
                  name="forecastHours"
                  value={formData.forecastHours}
                  onChange={handleInputChange}
                  min="6"
                  max="72"
                  step="6"
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>6h</span>
                  <span>24h</span>
                  <span>48h</span>
                  <span>72h</span>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading || apiStatus?.status !== 'healthy'}
                className="w-full bg-primary-600 text-white px-6 py-4 rounded-lg hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-semibold flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <Loader className="h-5 w-5 animate-spin" />
                    <span>Getting Prediction...</span>
                  </>
                ) : (
                  <>
                    <Cloud className="h-5 w-5" />
                    <span>Get Weather Prediction</span>
                  </>
                )}
              </button>
            </form>

            {/* Error Display */}
            {error && (
              <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
                <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-red-800">Error</h3>
                  <p className="text-sm text-red-700">{error}</p>
                  <p className="text-xs text-red-600 mt-1">
                    Make sure the backend is running at http://localhost:8000
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Results Display */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Prediction Results</h2>
            
            {!prediction && !loading && (
              <div className="flex flex-col items-center justify-center h-64 text-gray-400">
                <Cloud className="h-20 w-20 mb-4" />
                <p className="text-lg">No prediction yet</p>
                <p className="text-sm">Fill the form and click "Get Weather Prediction"</p>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center h-64">
                <Loader className="h-12 w-12 animate-spin text-primary-600 mb-4" />
                <p className="text-gray-600">Analyzing weather patterns...</p>
              </div>
            )}

            {prediction && (
              <div className="space-y-6">
                {/* Location Info with Hyperlocal Details */}
                <div className="bg-gradient-to-br from-blue-50 to-sky-50 rounded-xl p-5 border border-blue-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 text-primary-700 mb-2">
                        <MapPin className="h-5 w-5" />
                        <h3 className="font-semibold text-lg">{prediction.location}</h3>
                      </div>
                      {locationDetails && (
                        <div className="space-y-1 text-sm text-gray-700 mb-3">
                          {locationDetails.village && (
                            <p className="flex items-center space-x-2">
                              <span className="font-medium text-blue-700">🏘️ Village:</span>
                              <span>{locationDetails.village}</span>
                            </p>
                          )}
                          {locationDetails.district && (
                            <p className="flex items-center space-x-2">
                              <span className="font-medium text-blue-700">🏛️ District:</span>
                              <span>{locationDetails.district}</span>
                            </p>
                          )}
                          {locationDetails.state && (
                            <p className="flex items-center space-x-2">
                              <span className="font-medium text-blue-700">📍 State:</span>
                              <span>{locationDetails.state}</span>
                            </p>
                          )}
                        </div>
                      )}
                      <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                        <p>Lat: {prediction.latitude.toFixed(4)}</p>
                        <p>Lon: {prediction.longitude.toFixed(4)}</p>
                      </div>
                    </div>
                    <div className="bg-white rounded-lg px-3 py-2 text-xs font-medium text-blue-700 border border-blue-300">
                      {locationDetails?.village ? '🏞️ Hyperlocal' : '🏙️ City-Level'}
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-blue-200">
                    <p className="text-xs text-gray-600">
                      <span className="font-medium">Your Plan:</span> {formData.plannedActivity || formData.occupation}
                      {formData.activityTime && ` • ${activityTimes.find(t => t.value === formData.activityTime)?.label}`}
                      {formData.duration && ` • ${formData.duration}`}
                    </p>
                  </div>
                </div>

                {/* Weather Summary */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Weather Summary</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-red-50 rounded-lg p-3">
                      <div className="flex items-center space-x-2 text-red-700 mb-1">
                        <Cloud className="h-4 w-4" />
                        <span className="text-xs font-medium">Avg Temperature</span>
                      </div>
                      <p className="text-xl font-bold text-red-800">
                        {prediction.summary.avg_temperature.toFixed(1)}°C
                      </p>
                    </div>
                    <div className="bg-blue-50 rounded-lg p-3">
                      <div className="flex items-center space-x-2 text-blue-700 mb-1">
                        <Droplets className="h-4 w-4" />
                        <span className="text-xs font-medium">Total Rainfall</span>
                      </div>
                      <p className="text-xl font-bold text-blue-800">
                        {prediction.summary.total_rainfall.toFixed(1)}mm
                      </p>
                    </div>
                    <div className="bg-cyan-50 rounded-lg p-3">
                      <div className="flex items-center space-x-2 text-cyan-700 mb-1">
                        <Droplets className="h-4 w-4" />
                        <span className="text-xs font-medium">Avg Humidity</span>
                      </div>
                      <p className="text-xl font-bold text-cyan-800">
                        {prediction.summary.avg_humidity.toFixed(0)}%
                      </p>
                    </div>
                    <div className="bg-green-50 rounded-lg p-3">
                      <div className="flex items-center space-x-2 text-green-700 mb-1">
                        <Wind className="h-4 w-4" />
                        <span className="text-xs font-medium">Avg Wind Speed</span>
                      </div>
                      <p className="text-xl font-bold text-green-800">
                        {prediction.summary.avg_wind_speed.toFixed(1)} km/h
                      </p>
                    </div>
                  </div>
                </div>

                {/* Personalized Insight */}
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
                  <h3 className="font-bold text-purple-900 mb-3 flex items-center space-x-2">
                    <AlertCircle className="h-5 w-5" />
                    <span>Personalized Advisory</span>
                  </h3>
                  <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                    {prediction.personalized_insight}
                  </p>
                </div>

                {/* First 6 hours forecast */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Next 6 Hours Forecast</h3>
                  <div className="space-y-2">
                    {prediction.predictions.slice(0, 6).map((pred, idx) => (
                      <div key={idx} className="bg-gray-50 rounded-lg p-3 flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">
                          {new Date(pred.timestamp).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </span>
                        <div className="flex space-x-4 text-sm">
                          <span className="text-red-600 font-semibold">{pred.temperature.toFixed(1)}°C</span>
                          <span className="text-blue-600">{pred.humidity.toFixed(0)}%</span>
                          <span className="text-cyan-600">{pred.rainfall.toFixed(1)}mm</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Demo
