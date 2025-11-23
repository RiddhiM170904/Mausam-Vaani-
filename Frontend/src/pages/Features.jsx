import { Cloud, Database, Brain, MessageSquare, Globe, Code } from 'lucide-react'
import FeatureCard from '../components/FeatureCard'

const Features = () => {
  const detailedFeatures = [
    {
      icon: Cloud,
      title: 'Hyperlocal Forecasting',
      description: 'Advanced Temporal Fusion Transformer (TFT) models combined with real-time nowcasting provide street and village-scale micro-climate predictions.',
      innovation: 'Achieve unprecedented resolution with AI-powered forecasting that adapts to local terrain, urban heat islands, and micro-weather patterns.'
    },
    {
      icon: Database,
      title: 'Multi-Source Data Fusion',
      description: 'Integrate data from IMD, private weather APIs, low-cost IoT sensors, and crowdsourced observations to create a comprehensive weather picture.',
      innovation: 'Advanced quality control algorithms reconcile data anomalies and ensure reliability across diverse data sources.'
    },
    {
      icon: Brain,
      title: 'Impact Intelligence',
      description: 'Go beyond temperature and rainfall. Understand how weather affects agriculture, transportation, construction, and daily activities.',
      innovation: 'ML models translate weather data into sector-specific insights, helping you make informed decisions based on predicted outcomes.'
    },
    {
      icon: MessageSquare,
      title: 'AI-Powered Personalization',
      description: 'Gemini LLM analyzes user profiles, preferences, and predicted weather to generate personalized advisories in multiple languages.',
      innovation: 'Context-aware recommendations tailored to farmers, logistics managers, event planners, and everyday users.'
    },
    {
      icon: Globe,
      title: 'Universal Accessibility',
      description: 'Access weather intelligence through web, SMS, WhatsApp, and voice interfaces. Works on 2G networks to latest smartphones.',
      innovation: 'Truly inclusive design ensures everyone can access critical weather information regardless of technology access.'
    },
    {
      icon: Code,
      title: 'Developer API Ecosystem',
      description: 'Unified JSON API provides standardized weather advisory responses for seamless integration with government, enterprise, and third-party applications.',
      innovation: 'Enterprise-grade API with comprehensive documentation, SDKs, and developer support.'
    }
  ]

  const useCases = [
    {
      title: 'Agriculture',
      description: 'Optimize irrigation, predict pest outbreaks, and plan harvesting with hyperlocal weather data.',
      emoji: '🌾'
    },
    {
      title: 'Logistics',
      description: 'Route optimization and delay prediction based on real-time and forecasted weather conditions.',
      emoji: '🚚'
    },
    {
      title: 'Construction',
      description: 'Plan work schedules and protect materials with accurate precipitation and temperature forecasts.',
      emoji: '🏗️'
    },
    {
      title: 'Events',
      description: 'Make confident decisions about outdoor events with precise local weather predictions.',
      emoji: '🎪'
    },
    {
      title: 'Energy',
      description: 'Optimize renewable energy production with solar radiation and wind forecasts.',
      emoji: '⚡'
    },
    {
      title: 'Retail',
      description: 'Adjust inventory and staffing based on weather-influenced consumer behavior patterns.',
      emoji: '🛒'
    }
  ]

  return (
    <div className="min-h-screen py-12">
      {/* Header */}
      <section className="px-4 sm:px-6 lg:px-8 mb-16">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-6">
            Powerful Features for
            <span className="block text-primary-600 mt-2">Intelligent Weather Insights</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Our comprehensive platform combines cutting-edge AI, multi-source data fusion, and user-centric design 
            to deliver weather intelligence that truly makes a difference.
          </p>
        </div>
      </section>

      {/* Detailed Features */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-8">
            {detailedFeatures.map((feature, index) => (
              <FeatureCard key={index} {...feature} className="h-full" />
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20 bg-gradient-to-br from-gray-50 to-blue-50 py-16">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">
            Built on Cutting-Edge Technology
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white rounded-xl p-6 shadow-md">
              <h3 className="text-xl font-bold text-gray-900 mb-3">AI & ML</h3>
              <ul className="space-y-2 text-gray-600">
                <li>• Temporal Fusion Transformer</li>
                <li>• Gemini LLM Integration</li>
                <li>• Neural Weather Models</li>
                <li>• Anomaly Detection</li>
              </ul>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-md">
              <h3 className="text-xl font-bold text-gray-900 mb-3">Data Sources</h3>
              <ul className="space-y-2 text-gray-600">
                <li>• IMD Official Feeds</li>
                <li>• Private Weather APIs</li>
                <li>• IoT Sensor Networks</li>
                <li>• Crowdsourced Data</li>
              </ul>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-md">
              <h3 className="text-xl font-bold text-gray-900 mb-3">Delivery Channels</h3>
              <ul className="space-y-2 text-gray-600">
                <li>• Web Application</li>
                <li>• SMS Gateway</li>
                <li>• WhatsApp Bot</li>
                <li>• Voice Interface</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">
            Industry Use Cases
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {useCases.map((useCase, index) => (
              <div key={index} className="bg-white rounded-xl p-6 shadow-md hover:shadow-lg transition-shadow">
                <div className="text-4xl mb-4">{useCase.emoji}</div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{useCase.title}</h3>
                <p className="text-gray-600">{useCase.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto bg-primary-600 rounded-2xl p-12 text-center text-white shadow-2xl">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Transform Your Weather Intelligence?
          </h2>
          <p className="text-xl mb-8 text-primary-100">
            Experience the power of hyperlocal forecasting and AI-driven insights
          </p>
          <button className="bg-white text-primary-600 px-8 py-4 rounded-lg hover:bg-gray-100 transition-colors font-semibold text-lg shadow-lg">
            Request a Demo
          </button>
        </div>
      </section>
    </div>
  )
}

export default Features
