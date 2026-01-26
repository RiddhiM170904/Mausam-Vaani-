import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { MapPin, Cloud, Droplets, Wind, AlertCircle, Loader, User, Clock, Play, CheckCircle } from 'lucide-react'
import { GlassCard, Button } from '../components/Shared'
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
    activityTime: 'Morning (8-12 PM)',
    duration: '2-4 hours',
    concerns: [],
    crop: '',
    vehicle: '',
    forecastHours: 24,
  })
  const [locationDetails, setLocationDetails] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    checkApiHealth()
  }, [])

  const detectMyLocation = async () => {
    setDetectingLocation(true)
    setError(null)

    try {
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: false,
          timeout: 30000,
          maximumAge: 60000,
        })
      })

      const lat = position.coords.latitude
      const lon = position.coords.longitude

      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=18&addressdetails=1`
      )
      
      if (!response.ok) throw new Error('Failed to get location details')
      
      const data = await response.json()
      const address = data.address || {}

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

  const handleConcernToggle = (concernValue) => {
    setFormData(prev => ({
      ...prev,
      concerns: prev.concerns.includes(concernValue)
        ? prev.concerns.filter(c => c !== concernValue)
        : [...prev.concerns, concernValue]
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const additionalContext = {
        planned_activity: formData.plannedActivity || 'General activity',
        activity_time: formData.activityTime,
        duration: formData.duration,
        specific_concerns: formData.concerns.join(', '),
        location_type: locationDetails?.village ? 'Village' : 'City',
        village: locationDetails?.village || formData.village,
        district: locationDetails?.district || formData.district,
        state: locationDetails?.state || ''
      }

      if (formData.occupation === 'Farmer' && formData.crop) {
        additionalContext.crop = formData.crop
      }

      if (['Commuter', 'Delivery', 'General'].includes(formData.occupation) && formData.vehicle) {
        additionalContext.vehicle = formData.vehicle
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
    { value: 'Farmer', label: '🌾 Farmer' },
    { value: 'Commuter', label: '🚗 Commuter' },
    { value: 'Construction Worker', label: '👷 Construction' },
    { value: 'Outdoor Sports', label: '⚽ Sports' },
    { value: 'Student', label: '🎓 Student' },
    { value: 'Delivery', label: '📦 Delivery' },
    { value: 'Event Planner', label: '🎉 Events' },
    { value: 'General', label: '👤 General' }
  ]

  return (
    <div className="min-h-screen py-12">
      {/* Hero Section */}
      <section className="px-4 sm:px-6 lg:px-8 mb-12">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="text-6xl mb-6 block">🔮</span>
            <h1 className="text-4xl sm:text-5xl font-display font-bold text-white mb-6">
              Live <span className="neon-text">Weather Demo</span>
            </h1>
            <p className="text-xl text-white/60 max-w-3xl mx-auto">
              Experience personalized weather intelligence powered by AI. Get predictions tailored to your profession and activities.
            </p>
          </motion.div>

          {/* API Status */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="mt-6"
          >
            <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm ${
              apiStatus?.status === 'healthy' 
                ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                : 'bg-weather-storm/20 text-weather-storm border border-weather-storm/30'
            }`}>
              <span className={`w-2 h-2 rounded-full ${apiStatus?.status === 'healthy' ? 'bg-green-400' : 'bg-weather-storm'} animate-pulse`} />
              API Status: {apiStatus?.status === 'healthy' ? 'Online' : 'Offline'}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Main Content */}
      <section className="px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Input Form */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <GlassCard className="p-6">
                <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                  <User className="w-5 h-5 text-primary-400" />
                  Your Details
                </h2>

                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Location */}
                  <div className="space-y-3">
                    <label className="block text-sm font-medium text-white/70">Location</label>
                    <div className="flex gap-3">
                      <input
                        type="text"
                        name="locationName"
                        value={formData.locationName}
                        onChange={handleInputChange}
                        placeholder="Enter location or detect"
                        className="flex-1 px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:border-primary-500 outline-none transition-all"
                      />
                      <Button
                        type="button"
                        variant="secondary"
                        icon={MapPin}
                        onClick={detectMyLocation}
                        loading={detectingLocation}
                      >
                        Detect
                      </Button>
                    </div>
                    {locationDetails && (
                      <p className="text-xs text-primary-400">
                        📍 {locationDetails.fullAddress}
                      </p>
                    )}
                  </div>

                  {/* Occupation */}
                  <div className="space-y-3">
                    <label className="block text-sm font-medium text-white/70">Your Profession</label>
                    <div className="grid grid-cols-4 gap-2">
                      {occupations.map((occ) => (
                        <button
                          key={occ.value}
                          type="button"
                          onClick={() => setFormData(prev => ({ ...prev, occupation: occ.value }))}
                          className={`p-3 rounded-xl text-xs font-medium transition-all ${
                            formData.occupation === occ.value
                              ? 'bg-primary-500/30 border-2 border-primary-500 text-primary-300'
                              : 'bg-white/5 border border-white/10 text-white/60 hover:bg-white/10'
                          }`}
                        >
                          {occ.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Activity */}
                  <div className="space-y-3">
                    <label className="block text-sm font-medium text-white/70">Planned Activity</label>
                    <input
                      type="text"
                      name="plannedActivity"
                      value={formData.plannedActivity}
                      onChange={handleInputChange}
                      placeholder="e.g., Going for a morning walk"
                      className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:border-primary-500 outline-none transition-all"
                    />
                  </div>

                  {/* Time */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-white/70">Time</label>
                      <select
                        name="activityTime"
                        value={formData.activityTime}
                        onChange={handleInputChange}
                        className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-primary-500 outline-none transition-all"
                      >
                        <option className="bg-dark-800">Morning (6-12 AM)</option>
                        <option className="bg-dark-800">Afternoon (12-4 PM)</option>
                        <option className="bg-dark-800">Evening (4-8 PM)</option>
                        <option className="bg-dark-800">Night (8 PM-6 AM)</option>
                      </select>
                    </div>
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-white/70">Forecast Hours</label>
                      <select
                        name="forecastHours"
                        value={formData.forecastHours}
                        onChange={handleInputChange}
                        className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-primary-500 outline-none transition-all"
                      >
                        <option value="6" className="bg-dark-800">6 hours</option>
                        <option value="12" className="bg-dark-800">12 hours</option>
                        <option value="24" className="bg-dark-800">24 hours</option>
                        <option value="48" className="bg-dark-800">48 hours</option>
                      </select>
                    </div>
                  </div>

                  {error && (
                    <div className="p-4 rounded-xl bg-weather-storm/20 border border-weather-storm/30 text-weather-storm flex items-center gap-2">
                      <AlertCircle className="w-5 h-5" />
                      {error}
                    </div>
                  )}

                  <Button
                    type="submit"
                    variant="primary"
                    size="lg"
                    icon={Play}
                    loading={loading}
                    disabled={!formData.locationName}
                    className="w-full"
                  >
                    Get AI Weather Prediction
                  </Button>
                </form>
              </GlassCard>
            </motion.div>

            {/* Results */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <GlassCard className="p-6 h-full">
                <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                  <Cloud className="w-5 h-5 text-primary-400" />
                  AI Prediction
                </h2>

                {!prediction && !loading && (
                  <div className="flex flex-col items-center justify-center h-64 text-center">
                    <div className="w-20 h-20 rounded-full bg-primary-500/20 flex items-center justify-center mb-4">
                      <Cloud className="w-10 h-10 text-primary-400" />
                    </div>
                    <p className="text-white/40">
                      Enter your details and click "Get AI Weather Prediction" to see personalized forecasts
                    </p>
                  </div>
                )}

                {loading && (
                  <div className="flex flex-col items-center justify-center h-64">
                    <Loader className="w-12 h-12 text-primary-400 animate-spin mb-4" />
                    <p className="text-white/60">Generating personalized forecast...</p>
                  </div>
                )}

                {prediction && (
                  <div className="space-y-6">
                    {/* Current Weather */}
                    <div className="p-4 rounded-xl bg-gradient-to-br from-primary-500/20 to-primary-500/5 border border-primary-500/30">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <p className="text-white/60 text-sm">Current Weather</p>
                          <p className="text-3xl font-bold text-white">
                            {prediction.current_weather?.temperature || '28'}°C
                          </p>
                        </div>
                        <div className="text-5xl">
                          {prediction.current_weather?.condition?.includes('rain') ? '🌧️' : '☀️'}
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div className="flex items-center gap-2">
                          <Droplets className="w-4 h-4 text-weather-rain" />
                          <span className="text-white/70">{prediction.current_weather?.humidity || '65'}%</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Wind className="w-4 h-4 text-primary-400" />
                          <span className="text-white/70">{prediction.current_weather?.wind_speed || '12'} km/h</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Cloud className="w-4 h-4 text-white/60" />
                          <span className="text-white/70">{prediction.current_weather?.condition || 'Partly Cloudy'}</span>
                        </div>
                      </div>
                    </div>

                    {/* AI Advisory */}
                    {prediction.personalized_advisory && (
                      <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                        <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                          <CheckCircle className="w-4 h-4 text-green-400" />
                          AI Advisory for {formData.occupation}
                        </h3>
                        <p className="text-white/70 text-sm leading-relaxed">
                          {prediction.personalized_advisory}
                        </p>
                      </div>
                    )}

                    {/* Key Insights */}
                    {prediction.key_insights && prediction.key_insights.length > 0 && (
                      <div>
                        <h3 className="font-semibold text-white mb-3">Key Insights</h3>
                        <div className="space-y-2">
                          {prediction.key_insights.slice(0, 3).map((insight, index) => (
                            <div key={index} className="flex items-start gap-2 text-sm">
                              <span className="text-primary-400">•</span>
                              <span className="text-white/70">{insight}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </GlassCard>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Demo
