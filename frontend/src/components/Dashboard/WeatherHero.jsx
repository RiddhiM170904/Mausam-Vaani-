import { motion } from 'framer-motion'
import { Sun, Cloud, CloudRain, CloudSnow, CloudLightning, Wind, Droplets } from 'lucide-react'

// Map weather conditions to icons
const weatherIcons = {
  sunny: Sun,
  cloudy: Cloud,
  rain: CloudRain,
  snow: CloudSnow,
  storm: CloudLightning,
  windy: Wind,
}

const WeatherHero = ({ 
  weather = {
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
  }
}) => {
  const WeatherIcon = weatherIcons[weather.condition] || Sun

  const getConditionStyles = () => {
    switch (weather.condition) {
      case 'sunny':
        return {
          gradient: 'from-weather-sunny/20 to-orange-500/10',
          iconClass: 'text-weather-sunny icon-glow-sunny',
          bgAnimation: 'animate-pulse-slow'
        }
      case 'rain':
        return {
          gradient: 'from-weather-rain/20 to-blue-500/10',
          iconClass: 'text-weather-rain icon-glow-rain',
          bgAnimation: ''
        }
      case 'storm':
        return {
          gradient: 'from-weather-storm/20 to-purple-500/10',
          iconClass: 'text-weather-storm',
          bgAnimation: ''
        }
      default:
        return {
          gradient: 'from-primary-500/20 to-blue-500/10',
          iconClass: 'text-primary-400 icon-glow-primary',
          bgAnimation: ''
        }
    }
  }

  const styles = getConditionStyles()

  // AQI color coding
  const getAqiColor = (aqi) => {
    if (aqi <= 50) return 'text-green-400 bg-green-500/20'
    if (aqi <= 100) return 'text-yellow-400 bg-yellow-500/20'
    if (aqi <= 150) return 'text-orange-400 bg-orange-500/20'
    if (aqi <= 200) return 'text-red-400 bg-red-500/20'
    return 'text-purple-400 bg-purple-500/20'
  }

  const getAqiLabel = (aqi) => {
    if (aqi <= 50) return 'Good'
    if (aqi <= 100) return 'Moderate'
    if (aqi <= 150) return 'Unhealthy for Sensitive'
    if (aqi <= 200) return 'Unhealthy'
    return 'Very Unhealthy'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative overflow-hidden rounded-3xl bg-gradient-to-br ${styles.gradient} backdrop-blur-xl border border-white/10 p-6 sm:p-8`}
    >
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        {weather.condition === 'rain' && (
          <>
            {Array.from({ length: 20 }).map((_, i) => (
              <div
                key={i}
                className="rain-drop"
                style={{
                  left: `${Math.random() * 100}%`,
                  height: `${15 + Math.random() * 20}px`,
                  animationDuration: `${0.5 + Math.random() * 0.5}s`,
                  animationDelay: `${Math.random() * 2}s`,
                }}
              />
            ))}
          </>
        )}
        
        {weather.condition === 'sunny' && (
          <motion.div
            className="absolute -top-20 -right-20 w-60 h-60 bg-weather-sunny/10 rounded-full blur-3xl"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 4, repeat: Infinity }}
          />
        )}
      </div>

      <div className="relative z-10">
        {/* Top Row - Location & Time */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-lg text-white/70 mb-1">Current Weather</h2>
            <p className="text-2xl font-display font-bold text-white">{weather.location}</p>
          </div>
          <div className={`px-3 py-1.5 rounded-full ${getAqiColor(weather.aqi)}`}>
            <span className="text-xs font-medium">AQI {weather.aqi}</span>
            <span className="text-xs ml-1 opacity-70">• {getAqiLabel(weather.aqi)}</span>
          </div>
        </div>

        {/* Main Weather Display */}
        <div className="flex flex-col sm:flex-row items-center gap-6 sm:gap-10 mb-8">
          {/* Weather Icon */}
          <motion.div
            className={`relative ${styles.bgAnimation}`}
            animate={{ y: [0, -5, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          >
            <WeatherIcon className={`w-28 h-28 sm:w-36 sm:h-36 ${styles.iconClass}`} strokeWidth={1.5} />
            <div className="absolute inset-0 blur-2xl opacity-30">
              <WeatherIcon className={`w-28 h-28 sm:w-36 sm:h-36 ${styles.iconClass}`} />
            </div>
          </motion.div>

          {/* Temperature */}
          <div className="text-center sm:text-left">
            <div className="flex items-start justify-center sm:justify-start">
              <span className="text-7xl sm:text-8xl font-display font-bold text-white">
                {weather.temperature}
              </span>
              <span className="text-3xl text-white/60 mt-2">°C</span>
            </div>
            <p className="text-xl text-white/70 mt-1">{weather.description}</p>
            <p className="text-sm text-white/50">
              Feels like <span className="text-white/70">{weather.realFeel}°C</span>
            </p>
          </div>

          {/* High/Low */}
          <div className="hidden sm:flex flex-col gap-2 ml-auto">
            <div className="flex items-center gap-2 text-weather-sunny">
              <span className="text-sm">H:</span>
              <span className="text-lg font-semibold">{weather.high}°</span>
            </div>
            <div className="flex items-center gap-2 text-weather-rain">
              <span className="text-sm">L:</span>
              <span className="text-lg font-semibold">{weather.low}°</span>
            </div>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-3 gap-4">
          <div className="glass-card p-4 text-center">
            <Droplets className="w-5 h-5 text-weather-rain mx-auto mb-2" />
            <p className="text-lg font-semibold text-white">{weather.humidity}%</p>
            <p className="text-xs text-white/50">Humidity</p>
          </div>
          <div className="glass-card p-4 text-center">
            <Wind className="w-5 h-5 text-primary-400 mx-auto mb-2" />
            <p className="text-lg font-semibold text-white">{weather.windSpeed} km/h</p>
            <p className="text-xs text-white/50">Wind Speed</p>
          </div>
          <div className="glass-card p-4 text-center sm:hidden">
            <div className="flex justify-center gap-3">
              <div>
                <p className="text-sm font-semibold text-weather-sunny">{weather.high}°</p>
                <p className="text-xs text-white/50">High</p>
              </div>
              <div className="w-px bg-white/10" />
              <div>
                <p className="text-sm font-semibold text-weather-rain">{weather.low}°</p>
                <p className="text-xs text-white/50">Low</p>
              </div>
            </div>
          </div>
          <div className="hidden sm:block glass-card p-4 text-center">
            <CloudRain className="w-5 h-5 text-weather-rain mx-auto mb-2" />
            <p className="text-lg font-semibold text-white">20%</p>
            <p className="text-xs text-white/50">Rain Chance</p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default WeatherHero
