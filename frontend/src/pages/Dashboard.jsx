import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { RefreshCw, Settings, MapPin } from 'lucide-react'
import { WeatherHero, AIInsightsList, HourlyScroll, DailyList } from '../components/Dashboard'
import { VoiceButton, Button } from '../components/Shared'
import { useWeather } from '../context/WeatherContext'
import { useUser } from '../context/UserContext'

const Dashboard = () => {
  const { 
    currentWeather, 
    hourlyForecast, 
    dailyForecast, 
    insights,
    isLoading, 
    fetchWeather,
    fetchInsights 
  } = useWeather()
  
  const { user, persona } = useUser()

  useEffect(() => {
    fetchInsights(persona)
  }, [persona])

  const handleRefresh = () => {
    fetchWeather()
    fetchInsights(persona)
  }

  return (
    <div className="min-h-screen pb-24">
      {/* Dashboard Header */}
      <div className="sticky top-16 z-30 bg-dark-800/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <MapPin className="w-5 h-5 text-primary-400" />
              <div>
                <h1 className="text-lg font-semibold text-white">
                  {currentWeather?.location || 'Loading...'}
                </h1>
                <p className="text-xs text-white/50">
                  Last updated: {new Date().toLocaleTimeString()}
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Button 
                variant="ghost" 
                size="sm" 
                icon={RefreshCw}
                onClick={handleRefresh}
                className={isLoading ? 'animate-spin' : ''}
              >
                <span className="hidden sm:inline">Refresh</span>
              </Button>
              <Button variant="ghost" size="sm" icon={Settings}>
                <span className="hidden sm:inline">Settings</span>
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Weather Hero */}
            <WeatherHero weather={currentWeather} />

            {/* AI Insights */}
            <AIInsightsList insights={insights} persona={persona} />

            {/* Hourly Forecast */}
            <HourlyScroll hourlyData={hourlyForecast} />
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* 7-Day Forecast */}
            <DailyList dailyData={dailyForecast} />

            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button className="w-full p-3 rounded-xl bg-white/5 hover:bg-white/10 text-left text-white/70 hover:text-white transition-all flex items-center gap-3">
                  <span className="text-xl">🌾</span>
                  <span>View Crop Calendar</span>
                </button>
                <button className="w-full p-3 rounded-xl bg-white/5 hover:bg-white/10 text-left text-white/70 hover:text-white transition-all flex items-center gap-3">
                  <span className="text-xl">🗺️</span>
                  <span>Weather Radar Map</span>
                </button>
                <button className="w-full p-3 rounded-xl bg-white/5 hover:bg-white/10 text-left text-white/70 hover:text-white transition-all flex items-center gap-3">
                  <span className="text-xl">📊</span>
                  <span>Historical Data</span>
                </button>
                <button className="w-full p-3 rounded-xl bg-green-500/10 hover:bg-green-500/20 text-left text-green-400 hover:text-green-300 transition-all flex items-center gap-3 border border-green-500/20">
                  <span className="text-xl">💬</span>
                  <span>Share on WhatsApp</span>
                </button>
              </div>
            </motion.div>

            {/* Lite Mode Toggle */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="glass-card p-4"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white font-medium">Lite Mode</p>
                  <p className="text-xs text-white/50">For slow 2G/3G connections</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" className="sr-only peer" />
                  <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
                </label>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Voice Assistant FAB */}
      <VoiceButton />
    </div>
  )
}

export default Dashboard
