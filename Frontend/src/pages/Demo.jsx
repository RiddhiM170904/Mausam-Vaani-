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
          enableHighAccuracy: false, // Changed to false for faster response
          timeout: 30000, // Increased to 30 seconds
          maximumAge: 60000, // Use cached location if available within 1 minute
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
      // Build comprehensive user context for LLM
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

      // Add crop for farmers
      if (formData.occupation === 'Farmer' && formData.crop) {
        additionalContext.crop = formData.crop
      }

      // Add vehicle for commuters/delivery
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
    { value: 'Construction Worker', label: '👷 Construction Worker' },
    { value: 'Outdoor Sports', label: '⚽ Sports/Fitness' },
    { value: 'Student', label: '🎓 Student' },
    { value: 'Delivery', label: '📦 Delivery/Logistics' },
    { value: 'Event Planner', label: '🎉 Event Planner' },
    { value: 'Photographer', label: '📸 Photographer' },
    { value: 'Tourist', label: '✈️ Tourist/Traveler' },
    { value: 'General', label: '👤 General' }
  ]

  const activities = {
    'Farmer': [
      'Plowing/Tilling field',
      'Sowing seeds',
      'Pesticide spraying',
      'Harvesting crops',
      'Irrigation work',
      'Fertilizer application',
      'Other farming work'
    ],
    'Commuter': [
      'Office commute',
      'School/College commute',
      'Business travel',
      'Daily errands',
      'Two-wheeler ride',
      'Car drive',
      'Public transport travel'
    ],
    'Construction Worker': [
      'Building construction',
      'Road construction',
      'Painting/Plastering',
      'Roofing work',
      'Cement/Concrete work',
      'Excavation work',
      'Other construction work'
    ],
    'Outdoor Sports': [
      'Cricket match/practice',
      'Football match/practice',
      'Running/Jogging',
      'Cycling',
      'Outdoor gym/workout',
      'Yoga session',
      'Other sports activity'
    ],
    'Student': [
      'Going to school/college',
      'Outdoor classes',
      'Sports practice',
      'Field trip',
      'Exam day',
      'Group study outdoor',
      'Other school activity'
    ],
    'Delivery': [
      'Food delivery',
      'Package delivery',
      'E-commerce pickup/drop',
      'Document delivery',
      'Multiple deliveries',
      'Long-distance delivery'
    ],
    'Event Planner': [
      'Wedding ceremony',
      'Birthday celebration',
      'Corporate event',
      'Cultural program',
      'Religious ceremony',
      'Reception/Party',
      'Other event'
    ],
    'Photographer': [
      'Wedding shoot',
      'Outdoor photoshoot',
      'Event coverage',
      'Nature photography',
      'Product shoot',
      'Video shooting',
      'Other photography work'
    ],
    'Tourist': [
      'Sightseeing',
      'City tour',
      'Historical places visit',
      'Nature/Adventure trip',
      'Shopping/Markets',
      'Beach/Hill station visit',
      'Other tourist activity'
    ],
    'General': [
      'Shopping',
      'Visiting family/friends',
      'Medical appointment',
      'Outdoor meeting',
      'Picnic/Outing',
      'Walking/Strolling',
      'Attending function',
      'Other activity'
    ]
  }

  const activityTimes = [
    { value: 'Early morning (5-8 AM)', label: '🌅 Early Morning (5-8 AM)' },
    { value: 'Morning (8-12 PM)', label: '☀️ Morning (8-12 PM)' },
    { value: 'Afternoon (12-5 PM)', label: '🌤️ Afternoon (12-5 PM)' },
    { value: 'Evening (5-8 PM)', label: '🌆 Evening (5-8 PM)' },
    { value: 'Night (8 PM onwards)', label: '🌙 Night (8 PM+)' },
    { value: 'All day', label: '⏰ All Day' },
  ]

  const durations = [
    '< 1 hour',
    '1-2 hours',
    '2-4 hours',
    '4-6 hours',
    'Half day',
    'Full day',
    'Multiple days'
  ]

  const concernOptions = [
    { value: 'rain', label: '🌧️ Rain', emoji: '☔' },
    { value: 'heat', label: '🌡️ Extreme heat', emoji: '🥵' },
    { value: 'cold', label: '❄️ Cold weather', emoji: '🥶' },
    { value: 'wind', label: '💨 Strong wind', emoji: '🌬️' },
    { value: 'humidity', label: '💧 High humidity', emoji: '💦' },
    { value: 'aqi', label: '😷 Air pollution', emoji: '🏭' },
    { value: 'visibility', label: '🌫️ Fog/visibility', emoji: '👁️' },
    { value: 'storm', label: '⛈️ Storm/thunderstorm', emoji: '⚡' }
  ]

  const crops = [
    'Rice/Paddy',
    'Wheat',
    'Cotton',
    'Sugarcane',
    'Maize/Corn',
    'Pulses (Dal)',
    'Vegetables',
    'Fruits',
    'Soybean',
    'Groundnut',
    'Mustard',
    'Millets',
    'Tea/Coffee',
    'Spices',
    'Other crop'
  ]

  const vehicleTypes = {
    'Commuter': ['Car', 'Bike/Scooter', 'Bicycle', 'Public transport', 'Walking'],
    'Delivery': ['Bike/Scooter', 'Bicycle', 'Car', 'Auto/Tempo', 'On foot'],
    'General': ['Car', 'Bike/Scooter', 'Bicycle', 'Public transport', 'Walking']
  }

  return (
    <div className="min-h-screen px-4 py-12 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-bold text-gray-900">
            Live Weather Prediction Demo
          </h1>
          <p className="text-lg text-gray-600">
            Experience hyperlocal weather intelligence with AI-powered personalized insights
          </p>
          
          {/* API Status */}
          <div className="inline-flex items-center px-4 py-2 mt-4 bg-white rounded-lg shadow-md">
            <div className={`w-3 h-3 rounded-full mr-2 ${apiStatus?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm font-medium">
              {apiStatus?.status === 'healthy' ? 'API Connected' : 'API Offline'}
            </span>
          </div>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Input Form */}
          <div className="p-8 bg-white shadow-xl rounded-2xl">
            <h2 className="mb-6 text-2xl font-bold text-gray-900">Plan Your Activity</h2>
            
            <form onSubmit={handleSubmit} className="space-y-5">
              {/* Auto-Detect Location */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  📍 Location (Hyperlocal)
                </label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={detectMyLocation}
                    disabled={detectingLocation}
                    className="flex items-center px-4 py-3 space-x-2 font-medium text-white transition-colors bg-blue-600 rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
                  >
                    {detectingLocation ? (
                      <>
                        <Loader className="w-4 h-4 animate-spin" />
                        <span>Detecting...</span>
                      </>
                    ) : (
                      <>
                        <MapPin className="w-4 h-4" />
                        <span>Detect My Location</span>
                      </>
                    )}
                  </button>
                  <div className="flex items-center text-xs text-gray-500">
                    Or enter manually below
                  </div>
                </div>
              </div>

              {/* Location Details Display */}
              {locationDetails && (
                <div className="p-4 text-sm border border-blue-200 rounded-lg bg-blue-50">
                  <div className="flex items-center mb-2 space-x-2 font-semibold text-blue-900">
                    <MapPin className="w-4 h-4" />
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
                    <p className="mt-2 text-xs text-blue-600">
                      Coordinates: {locationDetails.coordinates.lat.toFixed(4)}, {locationDetails.coordinates.lon.toFixed(4)}
                    </p>
                  </div>
                </div>
              )}

              {/* Manual Location Input */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  Or Enter Location Manually
                </label>
                <input
                  type="text"
                  name="locationName"
                  value={formData.locationName}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  placeholder="e.g., Delhi, Mumbai, Sehore"
                  required
                />
              </div>

              {/* Occupation */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  Who are you?
                </label>
                <select
                  name="occupation"
                  value={formData.occupation}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  {occupations.map(occ => (
                    <option key={occ.value} value={occ.value}>{occ.label}</option>
                  ))}
                </select>
              </div>

              {/* Crop selection for farmers */}
              {formData.occupation === 'Farmer' && (
                <div>
                  <label className="block mb-2 text-sm font-medium text-gray-700">
                    🌾 Your Crop (Optional)
                  </label>
                  <select
                    name="crop"
                    value={formData.crop}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  >
                    <option value="">Select crop</option>
                    {crops.map(crop => (
                      <option key={crop} value={crop}>{crop}</option>
                    ))}
                  </select>
                </div>
              )}

              {/* Vehicle selection for commuters/delivery */}
              {['Commuter', 'Delivery', 'General'].includes(formData.occupation) && (
                <div>
                  <label className="block mb-2 text-sm font-medium text-gray-700">
                    🚗 Your Vehicle (Optional)
                  </label>
                  <select
                    name="vehicle"
                    value={formData.vehicle}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  >
                    <option value="">Select vehicle</option>
                    {vehicleTypes[formData.occupation]?.map(vehicle => (
                      <option key={vehicle} value={vehicle}>{vehicle}</option>
                    ))}
                  </select>
                </div>
              )}

              {/* Planned Activity */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  🎯 What are you planning?
                </label>
                <select
                  name="plannedActivity"
                  value={formData.plannedActivity}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  <option value="">Select activity (optional)</option>
                  {activities[formData.occupation]?.map(activity => (
                    <option key={activity} value={activity}>{activity}</option>
                  ))}
                </select>
              </div>

              {/* Activity Time */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  ⏰ When?
                </label>
                <select
                  name="activityTime"
                  value={formData.activityTime}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  {activityTimes.map(time => (
                    <option key={time.value} value={time.value}>{time.label}</option>
                  ))}
                </select>
              </div>

              {/* Duration */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  ⏱️ Duration
                </label>
                <select
                  name="duration"
                  value={formData.duration}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  {durations.map(dur => (
                    <option key={dur} value={dur}>{dur}</option>
                  ))}
                </select>
              </div>

              {/* Weather Concerns - Multi-select */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  ⚠️ Weather Concerns (select all that apply)
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {concernOptions.map(concern => (
                    <button
                      key={concern.value}
                      type="button"
                      onClick={() => handleConcernToggle(concern.value)}
                      className={`px-3 py-2 rounded-lg border-2 transition-all text-sm font-medium flex items-center justify-center space-x-1 ${
                        formData.concerns.includes(concern.value)
                          ? 'border-primary-500 bg-primary-50 text-primary-700'
                          : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                      }`}
                    >
                      <span>{concern.emoji}</span>
                      <span className="text-xs">{concern.value}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Forecast Hours */}
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
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
                <div className="flex justify-between mt-1 text-xs text-gray-500">
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
                className="flex items-center justify-center w-full px-6 py-4 space-x-2 font-semibold text-white transition-colors rounded-lg bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" />
                    <span>Getting Prediction...</span>
                  </>
                ) : (
                  <>
                    <Cloud className="w-5 h-5" />
                    <span>Get Weather Prediction</span>
                  </>
                )}
              </button>
            </form>

            {/* Error Display */}
            {error && (
              <div className="flex items-start p-4 mt-6 space-x-3 border border-red-200 rounded-lg bg-red-50">
                <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-red-800">Error</h3>
                  <p className="text-sm text-red-700">{error}</p>
                  <p className="mt-1 text-xs text-red-600">
                    Make sure the backend is running at http://localhost:8000
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Results Display */}
          <div className="p-8 bg-white shadow-xl rounded-2xl">
            <h2 className="mb-6 text-2xl font-bold text-gray-900">Prediction Results</h2>
            
            {!prediction && !loading && (
              <div className="flex flex-col items-center justify-center h-64 text-gray-400">
                <Cloud className="w-20 h-20 mb-4" />
                <p className="text-lg">No prediction yet</p>
                <p className="text-sm">Fill the form and click "Get Weather Prediction"</p>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center h-64">
                <Loader className="w-12 h-12 mb-4 animate-spin text-primary-600" />
                <p className="text-gray-600">Analyzing weather patterns...</p>
              </div>
            )}

            {prediction && (
              <div className="space-y-6">
                {/* Location Header */}
                <div className="p-5 border border-blue-200 bg-gradient-to-br from-blue-50 to-sky-50 rounded-xl">
                  <div className="flex items-center mb-2 space-x-2 text-primary-700">
                    <MapPin className="w-5 h-5" />
                    <h3 className="text-xl font-bold">{prediction.location}</h3>
                    <span className="px-2 py-1 text-xs font-medium text-blue-700 bg-white border border-blue-300 rounded">
                      {locationDetails?.village ? '🏞️ Hyperlocal' : '🏙️ City'}
                    </span>
                  </div>
                  {locationDetails && (
                    <div className="flex flex-wrap gap-3 text-sm text-gray-700">
                      {locationDetails.village && <span>🏘️ {locationDetails.village}</span>}
                      {locationDetails.district && <span>• 🏛️ {locationDetails.district}</span>}
                      {locationDetails.state && <span>• 📍 {locationDetails.state}</span>}
                    </div>
                  )}
                  <div className="pt-3 mt-3 border-t border-blue-200">
                    <p className="text-sm text-gray-600">
                      <span className="font-semibold">👤 {formData.occupation}</span>
                      {formData.plannedActivity && ` • ${formData.plannedActivity}`}
                      {formData.activityTime && ` • ${activityTimes.find(t => t.value === formData.activityTime)?.label}`}
                      {formData.duration && ` • ${formData.duration}`}
                      {formData.crop && ` • 🌾 ${formData.crop}`}
                      {formData.vehicle && ` • 🚗 ${formData.vehicle}`}
                    </p>
                  </div>
                </div>

                {/* Current Weather - Large Display */}
                {prediction.current_weather && (
                  <div className="p-8 text-center border-2 border-blue-300 bg-gradient-to-br from-blue-100 via-sky-50 to-blue-100 rounded-2xl">
                    <div className="mb-2 font-bold text-gray-900 text-7xl">
                      {prediction.current_weather.temp?.toFixed(0) || 'N/A'}°C
                    </div>
                    <div className="mb-4 text-xl font-medium text-gray-700 capitalize">
                      {prediction.current_weather.description || 'Clear'}
                    </div>
                    <div className="flex justify-center space-x-6 text-sm text-gray-600">
                      <span>🌡️ Feels {(prediction.current_weather.temp + (prediction.current_weather.humidity - 50) * 0.1).toFixed(0)}°C</span>
                      <span>💧 {prediction.current_weather.humidity}%</span>
                      <span>💨 {prediction.current_weather.wind_speed?.toFixed(1)} km/h</span>
                    </div>
                  </div>
                )}

                {/* All Weather Parameters Grid */}
                <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
                  {/* Temperature */}
                  <div className="p-4 border border-red-200 rounded-lg bg-gradient-to-br from-red-50 to-orange-50">
                    <div className="text-xs font-medium text-red-700">🌡️ Feels Like</div>
                    <div className="mt-1 text-2xl font-bold text-red-900">
                      {prediction.current_weather?.temp?.toFixed(0) || prediction.summary.avg_temperature.toFixed(0)}°C
                    </div>
                    <div className="mt-1 text-xs text-red-600">
                      {prediction.summary.min_temperature.toFixed(0)}° ~ {prediction.summary.max_temperature.toFixed(0)}°
                    </div>
                  </div>

                  {/* Wind */}
                  <div className="p-4 border border-green-200 rounded-lg bg-gradient-to-br from-green-50 to-emerald-50">
                    <div className="text-xs font-medium text-green-700">💨 Wind</div>
                    <div className="mt-1 text-2xl font-bold text-green-900">
                      {prediction.current_weather?.wind_speed?.toFixed(1) || prediction.summary.avg_wind_speed.toFixed(1)}
                    </div>
                    <div className="mt-1 text-xs text-green-600">km/h</div>
                  </div>

                  {/* Humidity */}
                  <div className="p-4 border border-blue-200 rounded-lg bg-gradient-to-br from-blue-50 to-cyan-50">
                    <div className="text-xs font-medium text-blue-700">💧 Humidity</div>
                    <div className="mt-1 text-2xl font-bold text-blue-900">
                      {prediction.current_weather?.humidity || prediction.summary.avg_humidity.toFixed(0)}%
                    </div>
                    <div className="mt-1 text-xs text-blue-600">Moisture</div>
                  </div>

                  {/* Pressure */}
                  <div className="p-4 border border-purple-200 rounded-lg bg-gradient-to-br from-purple-50 to-pink-50">
                    <div className="text-xs font-medium text-purple-700">🔘 Pressure</div>
                    <div className="mt-1 text-2xl font-bold text-purple-900">
                      {prediction.current_weather?.pressure?.toLocaleString() || '1,013'} hPa
                    </div>
                    <div className="mt-1 text-xs text-purple-600">Atmospheric</div>
                  </div>

                  {/* Visibility */}
                  <div className="p-4 border border-indigo-200 rounded-lg bg-gradient-to-br from-indigo-50 to-blue-50">
                    <div className="text-xs font-medium text-indigo-700">👁️ Visibility</div>
                    <div className="mt-1 text-2xl font-bold text-indigo-900">
                      {(prediction.current_weather?.visibility / 1000)?.toFixed(0) || '10'} km
                    </div>
                    <div className="mt-1 text-xs text-indigo-600">Clear</div>
                  </div>

                  {/* Cloud Cover */}
                  <div className="p-4 border border-gray-200 rounded-lg bg-gradient-to-br from-gray-50 to-slate-50">
                    <div className="text-xs font-medium text-gray-700">☁️ Clouds</div>
                    <div className="mt-1 text-2xl font-bold text-gray-900">
                      {prediction.current_weather?.cloud_cover || prediction.summary.avg_cloud_cover?.toFixed(0) || '0'}%
                    </div>
                    <div className="mt-1 text-xs text-gray-600">Cover</div>
                  </div>
                </div>

                {/* Air Quality Index */}
                {(prediction.predictions[0]?.aqi || prediction.current_weather?.aqi) && (
                  <div className="p-5 border-2 border-orange-300 rounded-xl bg-gradient-to-br from-orange-50 to-yellow-50">
                    <h3 className="mb-3 text-lg font-bold text-orange-900">💨 Air Quality Index</h3>
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <div className="text-5xl font-bold text-orange-900">
                          {prediction.predictions[0]?.aqi?.toFixed(0) || prediction.current_weather?.aqi?.toFixed(0) || '50'}
                        </div>
                        <div className="mt-1 text-sm font-medium text-orange-700">
                          {(prediction.predictions[0]?.aqi || 50) > 100 ? 'Poor' : (prediction.predictions[0]?.aqi || 50) > 50 ? 'Moderate' : 'Good'}
                        </div>
                      </div>
                      <div className="w-32 h-3 overflow-hidden bg-gray-200 rounded-full">
                        <div 
                          className="h-full transition-all bg-gradient-to-r from-green-400 via-yellow-400 via-orange-400 to-red-500"
                          style={{width: `${Math.min((prediction.predictions[0]?.aqi || 50) / 2, 100)}%`}}
                        />
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-4 gap-3">
                      <div className="p-3 bg-white border border-orange-200 rounded-lg">
                        <div className="text-xs font-medium text-gray-600">PM2.5</div>
                        <div className="text-lg font-bold text-orange-900">{prediction.predictions[0]?.pm25?.toFixed(0) || '0'}</div>
                      </div>
                      <div className="p-3 bg-white border border-orange-200 rounded-lg">
                        <div className="text-xs font-medium text-gray-600">PM10</div>
                        <div className="text-lg font-bold text-orange-900">{prediction.predictions[0]?.pm10?.toFixed(0) || '0'}</div>
                      </div>
                      <div className="p-3 bg-white border border-orange-200 rounded-lg">
                        <div className="text-xs font-medium text-gray-600">CO</div>
                        <div className="text-lg font-bold text-orange-900">{prediction.predictions[0]?.co?.toFixed(0) || '0'}</div>
                      </div>
                      <div className="p-3 bg-white border border-orange-200 rounded-lg">
                        <div className="text-xs font-medium text-gray-600">SO2</div>
                        <div className="text-lg font-bold text-orange-900">{prediction.predictions[0]?.so2?.toFixed(0) || '0'}</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Astronomical Data */}
                {prediction.predictions[0]?.sunrise && (
                  <div className="p-5 border border-yellow-300 bg-gradient-to-br from-yellow-50 to-orange-50 rounded-xl">
                    <h3 className="mb-4 text-lg font-bold text-yellow-900">🌅 Sun & Moon</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center space-x-3">
                        <div className="text-3xl">🌅</div>
                        <div>
                          <div className="text-xs font-medium text-yellow-700">Sunrise</div>
                          <div className="text-xl font-bold text-yellow-900">{prediction.predictions[0].sunrise || 'N/A'}</div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="text-3xl">🌇</div>
                        <div>
                          <div className="text-xs font-medium text-orange-700">Sunset</div>
                          <div className="text-xl font-bold text-orange-900">{prediction.predictions[0].sunset || 'N/A'}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Hourly Forecast */}
                <div className="p-5 border border-gray-300 rounded-xl bg-gradient-to-br from-gray-50 to-slate-50">
                  <h3 className="mb-4 text-lg font-bold text-gray-900">📊 Hourly Forecast ({formData.forecastHours}h)</h3>
                  <div className="overflow-x-auto">
                    <div className="flex pb-2 space-x-4">
                      {prediction.predictions.slice(0, Math.min(12, prediction.predictions.length)).map((hour, idx) => {
                        const hourTime = new Date(hour.timestamp);
                        return (
                          <div key={idx} className="flex flex-col items-center flex-shrink-0 p-3 bg-white border border-gray-200 rounded-lg w-28">
                            <div className="text-xs font-medium text-gray-600">
                              {idx === 0 ? 'Now' : hourTime.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'})}
                            </div>
                            <div className="my-2 text-2xl">
                              {hour.rainfall > 5 ? '🌧️' : hour.cloud_cover > 70 ? '☁️' : hour.cloud_cover > 30 ? '⛅' : '☀️'}
                            </div>
                            <div className="text-xl font-bold text-gray-900">{hour.temperature.toFixed(0)}°</div>
                            <div className="mt-1 text-xs text-gray-500">💧 {hour.humidity.toFixed(0)}%</div>
                            <div className="text-xs text-gray-500">💨 {hour.wind_speed.toFixed(1)}</div>
                            {hour.rainfall > 0 && (
                              <div className="mt-1 text-xs font-medium text-blue-600">🌧️ {hour.rainfall.toFixed(1)}mm</div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Personalized Insight */}
                <div className="p-6 border-2 border-purple-300 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl">
                  <h3 className="flex items-center mb-3 space-x-2 text-xl font-bold text-purple-900">
                    <AlertCircle className="w-6 h-6" />
                    <span>🤖 AI Advisory for {formData.occupation}</span>
                  </h3>
                  <p className="text-base leading-relaxed text-gray-800 whitespace-pre-wrap">
                    {prediction.personalized_insight}
                  </p>
                </div>

                {/* First 6 hours forecast */}
                <div>
                  <h3 className="mb-3 font-semibold text-gray-900">Next 6 Hours Forecast</h3>
                  <div className="space-y-2">
                    {prediction.predictions.slice(0, 6).map((pred, idx) => (
                      <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                        <span className="text-sm font-medium text-gray-700">
                          {new Date(pred.timestamp).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </span>
                        <div className="flex space-x-4 text-sm">
                          <span className="font-semibold text-red-600">{pred.temperature.toFixed(1)}°C</span>
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
