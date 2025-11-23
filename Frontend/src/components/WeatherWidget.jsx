// WeatherWidget.jsx
// This component is ready for backend integration
// It can be used to display real-time weather data once the API is connected

import { Cloud, Droplets, Wind, Eye, Thermometer } from 'lucide-react'

const WeatherWidget = ({ data }) => {
  // Mock data structure - replace with real API data
  const weatherData = data || {
    location: 'Your Location',
    temperature: 28,
    condition: 'Partly Cloudy',
    humidity: 65,
    windSpeed: 12,
    visibility: 10,
    precipitation: 30
  }

  return (
    <div className="bg-gradient-to-br from-blue-400 to-blue-600 rounded-2xl p-6 text-white shadow-xl">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h3 className="text-xl font-semibold mb-1">{weatherData.location}</h3>
          <p className="text-blue-100 text-sm">
            {new Date().toLocaleDateString('en-US', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </p>
        </div>
        <Cloud className="h-12 w-12 opacity-80" />
      </div>

      <div className="mb-6">
        <div className="flex items-baseline">
          <span className="text-6xl font-bold">{weatherData.temperature}</span>
          <span className="text-3xl ml-1">°C</span>
        </div>
        <p className="text-lg text-blue-100 mt-2">{weatherData.condition}</p>
      </div>

      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-blue-400">
        <div className="flex items-center space-x-2">
          <Droplets className="h-5 w-5 text-blue-200" />
          <div>
            <p className="text-xs text-blue-200">Humidity</p>
            <p className="font-semibold">{weatherData.humidity}%</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Wind className="h-5 w-5 text-blue-200" />
          <div>
            <p className="text-xs text-blue-200">Wind</p>
            <p className="font-semibold">{weatherData.windSpeed} km/h</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Eye className="h-5 w-5 text-blue-200" />
          <div>
            <p className="text-xs text-blue-200">Visibility</p>
            <p className="font-semibold">{weatherData.visibility} km</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Thermometer className="h-5 w-5 text-blue-200" />
          <div>
            <p className="text-xs text-blue-200">Rain Chance</p>
            <p className="font-semibold">{weatherData.precipitation}%</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WeatherWidget
