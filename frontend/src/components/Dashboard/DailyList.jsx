import { motion } from 'framer-motion'
import { Sun, Cloud, CloudRain, CloudSnow, CloudLightning, Wind } from 'lucide-react'

const weatherIcons = {
  sunny: Sun,
  cloudy: Cloud,
  rain: CloudRain,
  snow: CloudSnow,
  storm: CloudLightning,
  windy: Wind,
}

const DailyList = ({ dailyData = null }) => {
  // Default 7-day forecast
  const defaultData = [
    { day: 'Today', date: 'Jan 26', high: 32, low: 22, condition: 'sunny', rainChance: 10 },
    { day: 'Mon', date: 'Jan 27', high: 30, low: 21, condition: 'cloudy', rainChance: 25 },
    { day: 'Tue', date: 'Jan 28', high: 28, low: 20, condition: 'rain', rainChance: 80 },
    { day: 'Wed', date: 'Jan 29', high: 26, low: 19, condition: 'rain', rainChance: 90 },
    { day: 'Thu', date: 'Jan 30', high: 27, low: 18, condition: 'cloudy', rainChance: 40 },
    { day: 'Fri', date: 'Jan 31', high: 29, low: 19, condition: 'sunny', rainChance: 10 },
    { day: 'Sat', date: 'Feb 1', high: 31, low: 20, condition: 'sunny', rainChance: 5 },
  ]

  const data = dailyData || defaultData

  // Calculate the range for the temperature bar
  const allTemps = data.flatMap(d => [d.high, d.low])
  const minTemp = Math.min(...allTemps)
  const maxTemp = Math.max(...allTemps)
  const range = maxTemp - minTemp

  const getBarPosition = (temp) => {
    return ((temp - minTemp) / range) * 100
  }

  const getConditionColor = (condition) => {
    switch (condition) {
      case 'sunny':
        return 'text-weather-sunny'
      case 'rain':
        return 'text-weather-rain'
      case 'storm':
        return 'text-weather-storm'
      default:
        return 'text-white/60'
    }
  }

  return (
    <div className="glass-card p-4 sm:p-6">
      <h3 className="text-lg font-semibold text-white mb-4">7-Day Forecast</h3>

      <div className="space-y-2">
        {data.map((day, index) => {
          const Icon = weatherIcons[day.condition] || Cloud
          const isToday = day.day === 'Today'

          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`flex items-center gap-4 p-3 rounded-xl transition-all ${
                isToday 
                  ? 'bg-primary-500/10 border border-primary-500/20' 
                  : 'hover:bg-white/5'
              }`}
            >
              {/* Day Name */}
              <div className="w-16 sm:w-20">
                <p className={`font-medium ${isToday ? 'text-primary-400' : 'text-white'}`}>
                  {day.day}
                </p>
                <p className="text-xs text-white/40">{day.date}</p>
              </div>

              {/* Weather Icon */}
              <div className="w-10">
                <Icon className={`w-6 h-6 ${getConditionColor(day.condition)}`} />
              </div>

              {/* Rain Chance */}
              <div className="w-12 text-center">
                {day.rainChance > 0 && (
                  <span className="text-xs text-weather-rain">
                    {day.rainChance}%
                  </span>
                )}
              </div>

              {/* Temperature Bar */}
              <div className="flex-1 hidden sm:flex items-center gap-3">
                <span className="text-sm text-weather-rain w-8 text-right">{day.low}°</span>
                
                <div className="flex-1 h-2 bg-white/10 rounded-full relative">
                  {/* Temperature range bar */}
                  <div
                    className="absolute h-full rounded-full bg-gradient-to-r from-weather-rain via-primary-400 to-weather-sunny"
                    style={{
                      left: `${getBarPosition(day.low)}%`,
                      width: `${getBarPosition(day.high) - getBarPosition(day.low)}%`,
                    }}
                  />
                </div>
                
                <span className="text-sm text-weather-sunny w-8">{day.high}°</span>
              </div>

              {/* Mobile Temperature */}
              <div className="sm:hidden flex items-center gap-2 ml-auto">
                <span className="text-sm text-weather-rain">{day.low}°</span>
                <span className="text-white/30">/</span>
                <span className="text-sm text-weather-sunny">{day.high}°</span>
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

export default DailyList
