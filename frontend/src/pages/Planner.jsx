import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Calendar, ChevronLeft, ChevronRight, Wheat, Droplets, 
  Sun, CloudRain, AlertTriangle, Check, Clock
} from 'lucide-react'
import { GlassCard, Button } from '../components/Shared'
import { useUser } from '../context/UserContext'

const Planner = () => {
  const { persona } = useUser()
  const [currentMonth, setCurrentMonth] = useState(new Date())
  const [selectedDate, setSelectedDate] = useState(null)

  // Generate calendar days
  const getDaysInMonth = (date) => {
    const year = date.getFullYear()
    const month = date.getMonth()
    const firstDay = new Date(year, month, 1)
    const lastDay = new Date(year, month + 1, 0)
    const daysInMonth = lastDay.getDate()
    const startingDay = firstDay.getDay()
    
    const days = []
    
    // Previous month days
    for (let i = 0; i < startingDay; i++) {
      days.push({ day: null, isCurrentMonth: false })
    }
    
    // Current month days
    for (let i = 1; i <= daysInMonth; i++) {
      days.push({ 
        day: i, 
        isCurrentMonth: true,
        weather: getWeatherForDay(i),
        isGoodDay: getWeatherForDay(i).rainChance < 40,
      })
    }
    
    return days
  }

  const getWeatherForDay = (day) => {
    // Mock weather data - in real app, fetch from API
    const weathers = [
      { condition: 'sunny', rainChance: 10, temp: { high: 32, low: 22 } },
      { condition: 'cloudy', rainChance: 30, temp: { high: 30, low: 21 } },
      { condition: 'rain', rainChance: 80, temp: { high: 28, low: 20 } },
    ]
    return weathers[day % 3]
  }

  const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December']

  const navigateMonth = (direction) => {
    setCurrentMonth(prev => {
      const newMonth = new Date(prev)
      newMonth.setMonth(prev.getMonth() + direction)
      return newMonth
    })
  }

  const days = getDaysInMonth(currentMonth)

  // Farming tasks based on weather
  const farmingTasks = [
    { date: 'Jan 28', task: 'Best day for fertilizer spray', icon: Droplets, status: 'optimal' },
    { date: 'Jan 29', task: 'Good for irrigation', icon: Droplets, status: 'good' },
    { date: 'Jan 30', task: 'Avoid outdoor work', icon: CloudRain, status: 'bad' },
    { date: 'Feb 1', task: 'Ideal for harvesting', icon: Wheat, status: 'optimal' },
  ]

  return (
    <div className="min-h-screen py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-display font-bold text-white mb-2">
            {persona === 'farmer' ? 'Farming Calendar' : 'Weather Planner'}
          </h1>
          <p className="text-white/60">
            Plan your activities based on upcoming weather conditions
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Calendar */}
          <div className="lg:col-span-2">
            <GlassCard className="p-6">
              {/* Calendar Header */}
              <div className="flex items-center justify-between mb-6">
                <button 
                  onClick={() => navigateMonth(-1)}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <ChevronLeft className="w-5 h-5 text-white/60" />
                </button>
                <h2 className="text-xl font-semibold text-white">
                  {monthNames[currentMonth.getMonth()]} {currentMonth.getFullYear()}
                </h2>
                <button 
                  onClick={() => navigateMonth(1)}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <ChevronRight className="w-5 h-5 text-white/60" />
                </button>
              </div>

              {/* Day Names */}
              <div className="grid grid-cols-7 gap-2 mb-2">
                {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                  <div key={day} className="text-center text-sm text-white/40 py-2">
                    {day}
                  </div>
                ))}
              </div>

              {/* Calendar Days */}
              <div className="grid grid-cols-7 gap-2">
                {days.map((day, index) => (
                  <motion.button
                    key={index}
                    onClick={() => day.day && setSelectedDate(day)}
                    whileHover={{ scale: day.day ? 1.05 : 1 }}
                    whileTap={{ scale: day.day ? 0.95 : 1 }}
                    disabled={!day.day}
                    className={`aspect-square rounded-xl p-1 transition-all relative ${
                      !day.day 
                        ? 'opacity-0 cursor-default' 
                        : day.isGoodDay
                          ? 'bg-green-500/10 border border-green-500/30 hover:bg-green-500/20'
                          : 'bg-weather-storm/10 border border-weather-storm/30 hover:bg-weather-storm/20'
                    } ${selectedDate?.day === day.day ? 'ring-2 ring-primary-500' : ''}`}
                  >
                    {day.day && (
                      <>
                        <span className={`text-sm font-medium ${day.isGoodDay ? 'text-green-400' : 'text-weather-storm'}`}>
                          {day.day}
                        </span>
                        <div className="absolute bottom-1 left-1/2 -translate-x-1/2">
                          {day.weather.rainChance > 50 ? (
                            <CloudRain className="w-3 h-3 text-weather-rain" />
                          ) : (
                            <Sun className="w-3 h-3 text-weather-sunny" />
                          )}
                        </div>
                      </>
                    )}
                  </motion.button>
                ))}
              </div>

              {/* Legend */}
              <div className="flex items-center justify-center gap-6 mt-6 pt-4 border-t border-white/10">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                  <span className="text-sm text-white/60">Good for work</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-weather-storm" />
                  <span className="text-sm text-white/60">Bad weather</span>
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Sidebar - Tasks */}
          <div className="space-y-6">
            {/* Recommended Tasks */}
            <GlassCard className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-primary-400" />
                Recommended Activities
              </h3>
              <div className="space-y-3">
                {farmingTasks.map((task, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`p-3 rounded-xl border ${
                      task.status === 'optimal' 
                        ? 'bg-green-500/10 border-green-500/30' 
                        : task.status === 'good'
                          ? 'bg-weather-sunny/10 border-weather-sunny/30'
                          : 'bg-weather-storm/10 border-weather-storm/30'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <task.icon className={`w-5 h-5 ${
                        task.status === 'optimal' 
                          ? 'text-green-400' 
                          : task.status === 'good'
                            ? 'text-weather-sunny'
                            : 'text-weather-storm'
                      }`} />
                      <div className="flex-1">
                        <p className="text-sm font-medium text-white">{task.task}</p>
                        <p className="text-xs text-white/50">{task.date}</p>
                      </div>
                      {task.status === 'optimal' && (
                        <Check className="w-4 h-4 text-green-400" />
                      )}
                      {task.status === 'bad' && (
                        <AlertTriangle className="w-4 h-4 text-weather-storm" />
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </GlassCard>

            {/* Selected Day Details */}
            {selectedDate && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <GlassCard className="p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">
                    {monthNames[currentMonth.getMonth()]} {selectedDate.day}
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-white/60">Condition</span>
                      <span className="text-white capitalize">{selectedDate.weather.condition}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-white/60">Rain Chance</span>
                      <span className={selectedDate.weather.rainChance > 50 ? 'text-weather-rain' : 'text-green-400'}>
                        {selectedDate.weather.rainChance}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-white/60">Temperature</span>
                      <span className="text-white">
                        {selectedDate.weather.temp.low}° - {selectedDate.weather.temp.high}°
                      </span>
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Planner
