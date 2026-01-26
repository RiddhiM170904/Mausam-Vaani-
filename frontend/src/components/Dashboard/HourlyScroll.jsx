import { useRef } from 'react'
import { motion } from 'framer-motion'
import { ChevronLeft, ChevronRight, Sun, Cloud, CloudRain, CloudSnow, Moon } from 'lucide-react'

const weatherIcons = {
  sunny: Sun,
  cloudy: Cloud,
  rain: CloudRain,
  snow: CloudSnow,
  night: Moon,
}

const HourlyScroll = ({ hourlyData = null }) => {
  const scrollRef = useRef(null)

  // Default hourly data
  const defaultData = [
    { time: 'Now', temp: 32, condition: 'sunny', rainChance: 0 },
    { time: '1 PM', temp: 33, condition: 'sunny', rainChance: 5 },
    { time: '2 PM', temp: 34, condition: 'cloudy', rainChance: 10 },
    { time: '3 PM', temp: 33, condition: 'cloudy', rainChance: 20 },
    { time: '4 PM', temp: 31, condition: 'cloudy', rainChance: 35 },
    { time: '5 PM', temp: 29, condition: 'rain', rainChance: 60 },
    { time: '6 PM', temp: 27, condition: 'rain', rainChance: 75 },
    { time: '7 PM', temp: 26, condition: 'rain', rainChance: 50 },
    { time: '8 PM', temp: 25, condition: 'cloudy', rainChance: 25 },
    { time: '9 PM', temp: 24, condition: 'night', rainChance: 10 },
    { time: '10 PM', temp: 23, condition: 'night', rainChance: 5 },
    { time: '11 PM', temp: 22, condition: 'night', rainChance: 0 },
  ]

  const data = hourlyData || defaultData

  const scroll = (direction) => {
    if (scrollRef.current) {
      const scrollAmount = direction === 'left' ? -200 : 200
      scrollRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' })
    }
  }

  const getConditionStyle = (condition) => {
    switch (condition) {
      case 'sunny':
        return 'text-weather-sunny'
      case 'rain':
        return 'text-weather-rain'
      case 'night':
        return 'text-indigo-400'
      default:
        return 'text-white/60'
    }
  }

  return (
    <div className="glass-card p-4 sm:p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Hourly Forecast</h3>
        <div className="flex gap-2">
          <button
            onClick={() => scroll('left')}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
          >
            <ChevronLeft className="w-4 h-4 text-white/60" />
          </button>
          <button
            onClick={() => scroll('right')}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
          >
            <ChevronRight className="w-4 h-4 text-white/60" />
          </button>
        </div>
      </div>

      <div
        ref={scrollRef}
        className="flex gap-3 overflow-x-auto custom-scrollbar pb-2 -mx-2 px-2"
        style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
      >
        {data.map((hour, index) => {
          const Icon = weatherIcons[hour.condition] || Cloud
          const isNow = hour.time === 'Now'

          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`flex-shrink-0 w-20 p-3 rounded-xl text-center transition-all ${
                isNow 
                  ? 'bg-primary-500/20 border border-primary-500/40' 
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <p className={`text-sm font-medium mb-2 ${isNow ? 'text-primary-400' : 'text-white/60'}`}>
                {hour.time}
              </p>
              
              <motion.div
                whileHover={{ scale: 1.2, rotate: 10 }}
                className="mb-2"
              >
                <Icon className={`w-8 h-8 mx-auto ${getConditionStyle(hour.condition)}`} />
              </motion.div>
              
              <p className="text-lg font-semibold text-white mb-1">{hour.temp}°</p>
              
              {hour.rainChance > 0 && (
                <div className="flex items-center justify-center gap-1">
                  <CloudRain className="w-3 h-3 text-weather-rain" />
                  <span className="text-xs text-weather-rain">{hour.rainChance}%</span>
                </div>
              )}
            </motion.div>
          )
        })}
      </div>

      {/* Hide scrollbar but keep functionality */}
      <style jsx>{`
        div::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </div>
  )
}

export default HourlyScroll
