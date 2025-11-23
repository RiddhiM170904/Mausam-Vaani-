import { MapPin, Sparkles, TrendingUp, Users, Smartphone, Zap } from 'lucide-react'
import FeatureCard from '../components/FeatureCard'

const Home = () => {
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
      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-500 via-primary-600 to-blue-700 opacity-10"></div>
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
              Hyperlocal Weather Intelligence
              <span className="block text-primary-600 mt-2">For Everyone</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Empowering decisions with precise forecasts, actionable insights, and personalized advisories. 
              From street-level precision to multi-language accessibility.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="bg-primary-600 text-white px-8 py-4 rounded-lg hover:bg-primary-700 transition-colors font-semibold text-lg shadow-lg hover:shadow-xl">
                Get Started Free
              </button>
              <button className="bg-white text-primary-600 px-8 py-4 rounded-lg hover:bg-gray-50 transition-colors font-semibold text-lg border-2 border-primary-600">
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
