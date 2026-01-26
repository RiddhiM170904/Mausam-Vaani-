import { useState } from 'react'
import { motion } from 'framer-motion'
import { MapPin, Layers, Cloud, Wind, Droplets, Thermometer, ZoomIn, ZoomOut } from 'lucide-react'
import { GlassCard, Button } from '../components/Shared'

const Maps = () => {
  const [activeLayer, setActiveLayer] = useState('rain')
  const [zoom, setZoom] = useState(10)

  const layers = [
    { id: 'rain', label: 'Rain Radar', icon: Cloud, color: 'weather-rain' },
    { id: 'wind', label: 'Wind Speed', icon: Wind, color: 'primary' },
    { id: 'temp', label: 'Temperature', icon: Thermometer, color: 'weather-sunny' },
    { id: 'humidity', label: 'Humidity', icon: Droplets, color: 'weather-rain' },
  ]

  return (
    <div className="min-h-screen">
      {/* Map Container */}
      <div className="relative h-[calc(100vh-64px)]">
        {/* Placeholder Map - In real app, use Mapbox/Leaflet */}
        <div className="absolute inset-0 bg-dark-800">
          <div 
            className="w-full h-full bg-cover bg-center opacity-70"
            style={{
              backgroundImage: `url('https://api.mapbox.com/styles/v1/mapbox/dark-v11/static/78.9629,20.5937,5,0/1200x800?access_token=pk.placeholder')`,
              backgroundColor: '#1a1a2e',
            }}
          >
            {/* Simulated map with grid overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-dark-900/50 to-transparent" />
            
            {/* India outline placeholder */}
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="relative"
              >
                <div className="w-[400px] h-[400px] border border-primary-500/30 rounded-full opacity-20" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] border border-primary-500/30 rounded-full opacity-30" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[200px] h-[200px] border border-primary-500/30 rounded-full opacity-40" />
                
                {/* Weather overlay simulation */}
                {activeLayer === 'rain' && (
                  <>
                    <motion.div 
                      className="absolute top-20 left-20 w-24 h-24 bg-weather-rain/30 rounded-full blur-xl"
                      animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
                      transition={{ duration: 3, repeat: Infinity }}
                    />
                    <motion.div 
                      className="absolute bottom-32 right-24 w-32 h-32 bg-weather-rain/40 rounded-full blur-xl"
                      animate={{ scale: [1.1, 1, 1.1], opacity: [0.4, 0.6, 0.4] }}
                      transition={{ duration: 4, repeat: Infinity }}
                    />
                  </>
                )}
                
                {activeLayer === 'temp' && (
                  <>
                    <motion.div 
                      className="absolute top-16 right-16 w-28 h-28 bg-weather-sunny/30 rounded-full blur-xl"
                      animate={{ scale: [1, 1.15, 1], opacity: [0.3, 0.5, 0.3] }}
                      transition={{ duration: 3, repeat: Infinity }}
                    />
                    <motion.div 
                      className="absolute bottom-20 left-28 w-24 h-24 bg-orange-500/30 rounded-full blur-xl"
                      animate={{ scale: [1.1, 1, 1.1], opacity: [0.4, 0.5, 0.4] }}
                      transition={{ duration: 3.5, repeat: Infinity }}
                    />
                  </>
                )}
              </motion.div>
            </div>

            {/* Location Markers */}
            <motion.div 
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
              animate={{ y: [0, -5, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <div className="relative">
                <MapPin className="w-8 h-8 text-primary-400 fill-primary-400/30" />
                <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 whitespace-nowrap text-xs text-white/60">
                  Your Location
                </span>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Layer Controls */}
        <div className="absolute top-4 left-4 right-4 sm:right-auto z-10">
          <GlassCard className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-4 h-4 text-primary-400" />
              <span className="text-sm font-medium text-white">Layers</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {layers.map((layer) => (
                <button
                  key={layer.id}
                  onClick={() => setActiveLayer(layer.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all ${
                    activeLayer === layer.id
                      ? `bg-${layer.color}/20 border border-${layer.color}/40 text-${layer.color === 'primary' ? 'primary-400' : layer.color}`
                      : 'bg-white/5 border border-white/10 text-white/60 hover:bg-white/10'
                  }`}
                >
                  <layer.icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{layer.label}</span>
                </button>
              ))}
            </div>
          </GlassCard>
        </div>

        {/* Zoom Controls */}
        <div className="absolute bottom-20 right-4 z-10">
          <GlassCard className="p-2">
            <div className="flex flex-col gap-1">
              <button 
                onClick={() => setZoom(z => Math.min(z + 1, 15))}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <ZoomIn className="w-5 h-5 text-white/60" />
              </button>
              <div className="h-px bg-white/10" />
              <button 
                onClick={() => setZoom(z => Math.max(z - 1, 5))}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <ZoomOut className="w-5 h-5 text-white/60" />
              </button>
            </div>
          </GlassCard>
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 z-10">
          <GlassCard className="p-4">
            <p className="text-xs text-white/40 mb-2">
              {activeLayer === 'rain' && 'Precipitation Intensity'}
              {activeLayer === 'temp' && 'Temperature (°C)'}
              {activeLayer === 'wind' && 'Wind Speed (km/h)'}
              {activeLayer === 'humidity' && 'Humidity (%)'}
            </p>
            <div className="flex items-center gap-1">
              {activeLayer === 'rain' && (
                <>
                  <div className="w-6 h-3 bg-blue-200 rounded-sm" />
                  <div className="w-6 h-3 bg-blue-400 rounded-sm" />
                  <div className="w-6 h-3 bg-blue-600 rounded-sm" />
                  <div className="w-6 h-3 bg-purple-600 rounded-sm" />
                </>
              )}
              {activeLayer === 'temp' && (
                <>
                  <div className="w-6 h-3 bg-blue-500 rounded-sm" />
                  <div className="w-6 h-3 bg-green-500 rounded-sm" />
                  <div className="w-6 h-3 bg-yellow-500 rounded-sm" />
                  <div className="w-6 h-3 bg-red-500 rounded-sm" />
                </>
              )}
            </div>
            <div className="flex justify-between text-xs text-white/40 mt-1">
              <span>Low</span>
              <span>High</span>
            </div>
          </GlassCard>
        </div>

        {/* Location Info Card */}
        <div className="absolute bottom-4 right-4 z-10 hidden sm:block">
          <GlassCard className="p-4 w-64">
            <div className="flex items-center gap-2 mb-3">
              <MapPin className="w-4 h-4 text-primary-400" />
              <span className="text-sm font-medium text-white">Kothri Kalan, MP</span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-white/40">Lat</p>
                <p className="text-white">23.2599°N</p>
              </div>
              <div>
                <p className="text-white/40">Long</p>
                <p className="text-white">77.4126°E</p>
              </div>
              <div>
                <p className="text-white/40">Zoom</p>
                <p className="text-white">{zoom}x</p>
              </div>
              <div>
                <p className="text-white/40">Layer</p>
                <p className="text-white capitalize">{activeLayer}</p>
              </div>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  )
}

export default Maps
