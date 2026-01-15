import { MapPin, Sparkles, TrendingUp, Users, Smartphone, Zap, Cloud, CloudRain, Sun, Moon, Wind } from 'lucide-react'
import FeatureCard from '../components/FeatureCard'
import { useState, useEffect } from 'react'

const Home = () => {
  const [weatherMode, setWeatherMode] = useState('sunny') // sunny, rainy, night
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    // Auto-cycle through weather modes
    const interval = setInterval(() => {
      setWeatherMode(prev => {
        if (prev === 'sunny') return 'rainy'
        if (prev === 'rainy') return 'night'
        return 'sunny'
      })
    }, 10000) // Change every 10 seconds

    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  const features = [
    {
      icon: MapPin,
      title: 'Hyperlocal Forecasting',
      description: 'Temporal Fusion Transformer (TFT) + nowcasting integration',
      innovation: 'Street / village-scale micro-climate resolution'
    },
    {
      icon: Sparkles,
      title: 'Data Fusion',
      description: 'IMD feeds + private APIs + low-cost IoT + crowdsourcing',
      innovation: 'Robust QC & anomaly reconciliation'
    },
    {
      icon: TrendingUp,
      title: 'Impact Intelligence',
      description: 'Weather → Activity / Crop / Sector outcomes',
      innovation: 'Actionability beyond raw metrics'
    },
    {
      icon: Users,
      title: 'Personalization',
      description: 'Gemini LLM + user metadata + predicted weather',
      innovation: 'On-demand advisory (multi-role, multi-language)'
    },
    {
      icon: Smartphone,
      title: 'Accessibility',
      description: 'Web + SMS + WhatsApp + Voice',
      innovation: 'Inclusive reach (2G → smartphone)'
    },
    {
      icon: Zap,
      title: 'API Ecosystem',
      description: 'Unified JSON advisory responses',
      innovation: 'Government, enterprise & developer integrations'
    }
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section with Animated Weather Background */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8 overflow-hidden min-h-[600px]">
        {/* Animated Background */}
        <div className={`absolute inset-0 transition-all duration-1000 ${
          weatherMode === 'night' 
            ? 'bg-gradient-to-br from-indigo-900 via-purple-900 to-blue-900' 
            : weatherMode === 'rainy'
            ? 'bg-gradient-to-br from-gray-600 via-gray-700 to-blue-800'
            : 'bg-gradient-to-br from-sky-400 via-blue-400 to-cyan-300'
        }`}>
          
          {/* Sun */}
          {weatherMode === 'sunny' && (
            <div className="absolute top-10 right-20 animate-pulse">
              <div className="relative">
                <Sun className="w-24 h-24 text-yellow-300 animate-spin" style={{ animationDuration: '20s' }} />
                <div className="absolute inset-0 w-24 h-24 bg-yellow-200 rounded-full blur-xl opacity-50"></div>
              </div>
            </div>
          )}

          {/* Moon and Stars */}
          {weatherMode === 'night' && (
            <>
              <div className="absolute top-10 right-20">
                <Moon className="w-20 h-20 text-yellow-100" />
                <div className="absolute inset-0 w-20 h-20 bg-yellow-100 rounded-full blur-2xl opacity-30"></div>
              </div>
              {[...Array(30)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-1 h-1 bg-white rounded-full animate-pulse"
                  style={{
                    top: `${Math.random() * 60}%`,
                    left: `${Math.random() * 100}%`,
                    animationDelay: `${Math.random() * 2}s`,
                    animationDuration: `${2 + Math.random() * 2}s`
                  }}
                />
              ))}
            </>
          )}

          {/* Animated Clouds */}
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className={`absolute ${weatherMode === 'night' ? 'text-indigo-800' : 'text-white'} opacity-70`}
              style={{
                top: `${10 + i * 15}%`,
                left: '-10%',
                animation: `float ${20 + i * 5}s linear infinite`,
                animationDelay: `${i * 2}s`,
              }}
            >
              <Cloud className="w-32 h-32" />
            </div>
          ))}

          {/* Rain Effect */}
          {weatherMode === 'rainy' && (
            <div className="absolute inset-0">
              {[...Array(50)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-0.5 bg-blue-200 opacity-40"
                  style={{
                    left: `${Math.random() * 100}%`,
                    top: '-10%',
                    height: `${20 + Math.random() * 30}px`,
                    animation: `rain ${0.5 + Math.random() * 0.5}s linear infinite`,
                    animationDelay: `${Math.random() * 2}s`
                  }}
                />
              ))}
              <CloudRain className="absolute top-5 left-1/4 w-16 h-16 text-gray-300 animate-bounce" style={{ animationDuration: '3s' }} />
              <CloudRain className="absolute top-10 right-1/3 w-20 h-20 text-gray-300 animate-bounce" style={{ animationDuration: '2.5s' }} />
            </div>
          )}

          {/* Interactive Wind Effect */}
          <div 
            className="absolute opacity-20"
            style={{
              left: `${mousePosition.x / 10}px`,
              top: `${mousePosition.y / 10}px`,
              transition: 'all 0.5s ease-out'
            }}
          >
            <Wind className="w-40 h-40 text-white" />
          </div>
        </div>

        {/* Weather Mode Toggle Buttons */}
        <div className="absolute top-4 right-4 z-20 flex gap-2">
          <button
            onClick={() => setWeatherMode('sunny')}
            className={`p-2 rounded-full transition-all ${weatherMode === 'sunny' ? 'bg-yellow-400 scale-110' : 'bg-white/30'}`}
          >
            <Sun className="w-5 h-5 text-white" />
          </button>
          <button
            onClick={() => setWeatherMode('rainy')}
            className={`p-2 rounded-full transition-all ${weatherMode === 'rainy' ? 'bg-blue-500 scale-110' : 'bg-white/30'}`}
          >
            <CloudRain className="w-5 h-5 text-white" />
          </button>
          <button
            onClick={() => setWeatherMode('night')}
            className={`p-2 rounded-full transition-all ${weatherMode === 'night' ? 'bg-indigo-600 scale-110' : 'bg-white/30'}`}
          >
            <Moon className="w-5 h-5 text-white" />
          </button>
        </div>

        <style jsx>{`
          @keyframes float {
            0% {
              transform: translateX(0) translateY(0);
            }
            100% {
              transform: translateX(110vw) translateY(-20px);
            }
          }
          
          @keyframes rain {
            0% {
              transform: translateY(0);
            }
            100% {
              transform: translateY(100vh);
            }
          }
        `}</style>
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6 drop-shadow-lg animate-fadeIn">
              Hyperlocal Weather Intelligence
              <span className="block text-yellow-300 mt-2">For Everyone</span>
            </h1>
            <p className="text-xl text-white mb-8 max-w-3xl mx-auto drop-shadow-md">
              Empowering decisions with precise forecasts, actionable insights, and personalized advisories. 
              From street-level precision to multi-language accessibility.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="bg-white text-blue-600 px-8 py-4 rounded-lg hover:bg-blue-50 transition-all font-semibold text-lg shadow-2xl hover:shadow-xl hover:scale-105 transform">
                Get Started Free
              </button>
              <button className="bg-blue-600/30 backdrop-blur-md text-white px-8 py-4 rounded-lg hover:bg-blue-600/50 transition-all font-semibold text-lg border-2 border-white shadow-2xl hover:scale-105 transform">
                View Demo
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Weather Dashboard Preview */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="bg-gradient-to-br from-blue-50 to-sky-100 rounded-2xl shadow-2xl p-8 md:p-12">
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-white rounded-xl p-6 shadow-md">
                <div className="text-4xl mb-2">🌤️</div>
                <h3 className="text-lg font-semibold text-gray-900">Current Weather</h3>
                <p className="text-3xl font-bold text-primary-600 mt-2">28°C</p>
                <p className="text-sm text-gray-600">Partly Cloudy</p>
              </div>
              <div className="bg-white rounded-xl p-6 shadow-md">
                <div className="text-4xl mb-2">💧</div>
                <h3 className="text-lg font-semibold text-gray-900">Precipitation</h3>
                <p className="text-3xl font-bold text-blue-600 mt-2">30%</p>
                <p className="text-sm text-gray-600">Light rain expected</p>
              </div>
              <div className="bg-white rounded-xl p-6 shadow-md">
                <div className="text-4xl mb-2">🌾</div>
                <h3 className="text-lg font-semibold text-gray-900">Crop Advisory</h3>
                <p className="text-sm font-medium text-green-600 mt-2">Good for irrigation</p>
                <p className="text-sm text-gray-600">Optimal conditions</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Core Features */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
              Core Value Proposition
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Six layers of innovation delivering weather intelligence that transforms raw data into actionable insights
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <FeatureCard key={index} {...feature} />
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-primary-600 text-white">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold mb-2">99.9%</div>
              <div className="text-primary-100">Accuracy</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">10M+</div>
              <div className="text-primary-100">Users Served</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">500+</div>
              <div className="text-primary-100">Cities Covered</div>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">24/7</div>
              <div className="text-primary-100">Live Updates</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-6">
            Ready to Experience Hyperlocal Weather Intelligence?
          </h2>
          <p className="text-lg text-gray-600 mb-8">
            Join thousands of users who trust Mausam Vaani for precise, actionable weather insights
          </p>
          <button className="bg-primary-600 text-white px-10 py-4 rounded-lg hover:bg-primary-700 transition-colors font-semibold text-lg shadow-lg hover:shadow-xl">
            Start Your Free Trial
          </button>
        </div>
      </section>
    </div>
  )
}

export default Home
