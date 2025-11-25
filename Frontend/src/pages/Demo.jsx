import { useState, useEffect } from 'react'
import { MapPin, Cloud, Droplets, Wind, AlertCircle, Loader } from 'lucide-react'
import { getWeatherPrediction, checkHealth } from '../services/weatherApi'

const Demo = () => {
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState(null)
  const [formData, setFormData] = useState({
    latitude: 28.6139,
    longitude: 77.2090,
    locationName: 'Delhi',
    profession: 'Farmer',
    crop: 'Rice',
    forecastHours: 24,
  })
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  // Check API health on mount
  useEffect(() => {
    checkApiHealth()
  }, [])

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
      const result = await getWeatherPrediction({
        weatherInput: {
          latitude: parseFloat(formData.latitude),
          longitude: parseFloat(formData.longitude),
          location_name: formData.locationName,
        },
        userContext: {
          profession: formData.profession,
          additional_context: formData.profession === 'Farmer' ? { crop: formData.crop } : {},
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

  const professions = ['Farmer', 'Commuter', 'Construction Worker', 'Outdoor Sports', 'General']
  const crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize']

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
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Enter Your Details</h2>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Location */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Location Name
                </label>
                <input
                  type="text"
                  name="locationName"
                  value={formData.locationName}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                  placeholder="e.g., Delhi, Mumbai"
                />
              </div>

              {/* Coordinates */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Latitude
                  </label>
                  <input
                    type="number"
                    name="latitude"
                    value={formData.latitude}
                    onChange={handleInputChange}
                    step="0.0001"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Longitude
                  </label>
                  <input
                    type="number"
                    name="longitude"
                    value={formData.longitude}
                    onChange={handleInputChange}
                    step="0.0001"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                  />
                </div>
              </div>

              {/* Profession */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Your Profession
                </label>
                <select
                  name="profession"
                  value={formData.profession}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                >
                  {professions.map(prof => (
                    <option key={prof} value={prof}>{prof}</option>
                  ))}
                </select>
              </div>

              {/* Crop (conditional) */}
              {formData.profession === 'Farmer' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Crop Type
                  </label>
                  <select
                    name="crop"
                    value={formData.crop}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                  >
                    {crops.map(crop => (
                      <option key={crop} value={crop}>{crop}</option>
                    ))}
                  </select>
                </div>
              )}

              {/* Forecast Hours */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Forecast Hours: {formData.forecastHours}
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
                {/* Location Info */}
                <div className="bg-gradient-to-br from-blue-50 to-sky-50 rounded-xl p-4">
                  <div className="flex items-center space-x-2 text-primary-700 mb-2">
                    <MapPin className="h-5 w-5" />
                    <h3 className="font-semibold">{prediction.location}</h3>
                  </div>
                  <p className="text-sm text-gray-600">
                    {prediction.latitude.toFixed(4)}, {prediction.longitude.toFixed(4)}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Profession: {prediction.profession}
                  </p>
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
