import { motion } from 'framer-motion'
import { 
  Cloud, Database, Brain, MessageSquare, Globe, Code,
  Zap, Shield, Users, Smartphone, Mic, MessageCircle
} from 'lucide-react'
import { GlassCard, Button } from '../components/Shared'
import { Link } from 'react-router-dom'

const Features = () => {
  const coreFeatures = [
    {
      icon: Cloud,
      title: 'Hyperlocal Forecasting',
      description: 'Advanced Temporal Fusion Transformer (TFT) models combined with real-time nowcasting provide street and village-scale micro-climate predictions.',
      innovation: 'Achieve unprecedented resolution with AI-powered forecasting that adapts to local terrain, urban heat islands, and micro-weather patterns.',
      color: 'weather-rain'
    },
    {
      icon: Database,
      title: 'Multi-Source Data Fusion',
      description: 'Integrate data from IMD, private weather APIs, low-cost IoT sensors, and crowdsourced observations to create a comprehensive weather picture.',
      innovation: 'Advanced quality control algorithms reconcile data anomalies and ensure reliability across diverse data sources.',
      color: 'primary'
    },
    {
      icon: Brain,
      title: 'Impact Intelligence',
      description: 'Go beyond temperature and rainfall. Understand how weather affects agriculture, transportation, construction, and daily activities.',
      innovation: 'ML models translate weather data into sector-specific insights, helping you make informed decisions based on predicted outcomes.',
      color: 'weather-sunny'
    },
    {
      icon: MessageSquare,
      title: 'AI-Powered Personalization',
      description: 'Gemini LLM analyzes user profiles, preferences, and predicted weather to generate personalized advisories in multiple languages.',
      innovation: 'Context-aware recommendations tailored to farmers, logistics managers, event planners, and everyday users.',
      color: 'purple'
    },
    {
      icon: Globe,
      title: 'Universal Accessibility',
      description: 'Access weather intelligence through web, SMS, WhatsApp, and voice interfaces. Works on 2G networks to latest smartphones.',
      innovation: 'Truly inclusive design ensures everyone can access critical weather information regardless of technology access.',
      color: 'green'
    },
    {
      icon: Code,
      title: 'Developer API Ecosystem',
      description: 'Unified JSON API provides standardized weather advisory responses for seamless integration with government, enterprise, and third-party applications.',
      innovation: 'Enterprise-grade API with comprehensive documentation, SDKs, and developer support.',
      color: 'weather-storm'
    }
  ]

  const indianFeatures = [
    {
      icon: MessageCircle,
      title: 'WhatsApp Integration',
      description: 'Get daily forecasts and severe weather alerts directly on WhatsApp. Share with family and community groups instantly.',
      emoji: '💬'
    },
    {
      icon: Mic,
      title: 'Mausam Sahayak',
      description: 'Voice assistant that understands Hindi and regional languages. Ask weather questions naturally and get spoken responses.',
      emoji: '🎙️'
    },
    {
      icon: Users,
      title: 'Community Reports',
      description: 'Contribute and view real-time weather observations from your neighborhood. Verified by locals, for locals.',
      emoji: '👥'
    },
    {
      icon: Smartphone,
      title: 'Lite Mode',
      description: 'Optimized for 2G/3G connections. Get essential weather updates even with slow internet connectivity.',
      emoji: '📱'
    }
  ]

  const useCases = [
    { title: 'Agriculture', desc: 'Optimize irrigation and plan harvesting', emoji: '🌾' },
    { title: 'Logistics', desc: 'Route optimization and delay prediction', emoji: '🚚' },
    { title: 'Construction', desc: 'Plan work and protect materials', emoji: '🏗️' },
    { title: 'Events', desc: 'Confident outdoor event planning', emoji: '🎪' },
    { title: 'Energy', desc: 'Optimize renewable production', emoji: '⚡' },
    { title: 'Retail', desc: 'Weather-based inventory planning', emoji: '🛒' },
  ]

  const getColorClasses = (color) => {
    const colors = {
      'weather-rain': 'text-weather-rain bg-weather-rain/10 border-weather-rain/30',
      'primary': 'text-primary-400 bg-primary-500/10 border-primary-500/30',
      'weather-sunny': 'text-weather-sunny bg-weather-sunny/10 border-weather-sunny/30',
      'purple': 'text-purple-400 bg-purple-500/10 border-purple-500/30',
      'green': 'text-green-400 bg-green-500/10 border-green-500/30',
      'weather-storm': 'text-weather-storm bg-weather-storm/10 border-weather-storm/30',
    }
    return colors[color] || colors['primary']
  }

  return (
    <div className="min-h-screen py-12">
      {/* Hero Section */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="inline-block px-4 py-2 rounded-full bg-primary-500/10 text-primary-400 text-sm font-medium mb-6 border border-primary-500/20">
              <Zap className="inline w-4 h-4 mr-1" />
              Powerful Capabilities
            </span>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-display font-bold text-white mb-6">
              Features Built for
              <span className="block neon-text mt-2">Indian Weather Intelligence</span>
            </h1>
            <p className="text-xl text-white/60 max-w-3xl mx-auto">
              Our comprehensive platform combines cutting-edge AI, multi-source data fusion, 
              and user-centric design to deliver weather intelligence that truly makes a difference.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Core Features Grid */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <motion.h2 
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            className="text-2xl font-semibold text-white mb-8 flex items-center gap-2"
          >
            <Shield className="w-6 h-6 text-primary-400" />
            Core Technology
          </motion.h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            {coreFeatures.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlassCard className="h-full p-6 hover:scale-[1.02] transition-transform">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 border ${getColorClasses(feature.color)}`}>
                    <feature.icon className="w-6 h-6" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                  <p className="text-white/70 mb-4 leading-relaxed">{feature.description}</p>
                  <div className="pt-4 border-t border-white/10">
                    <p className="text-sm text-primary-400/80">
                      <span className="font-medium">Innovation:</span> {feature.innovation}
                    </p>
                  </div>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* India-Specific Features */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            className="text-center mb-12"
          >
            <span className="text-4xl mb-4 block">🇮🇳</span>
            <h2 className="text-3xl font-bold text-white mb-4">Made for India</h2>
            <p className="text-white/60 max-w-2xl mx-auto">
              Features designed specifically for Indian users, networks, and languages
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {indianFeatures.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlassCard className="h-full p-6 text-center">
                  <span className="text-4xl mb-4 block">{feature.emoji}</span>
                  <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                  <p className="text-sm text-white/60">{feature.description}</p>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <motion.h2 
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            className="text-2xl font-semibold text-white mb-8 text-center"
          >
            Industry Applications
          </motion.h2>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {useCases.map((useCase, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="glass-card p-4 text-center cursor-pointer"
              >
                <span className="text-3xl mb-2 block">{useCase.emoji}</span>
                <h3 className="font-medium text-white text-sm mb-1">{useCase.title}</h3>
                <p className="text-xs text-white/50">{useCase.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <GlassCard className="p-8 md:p-12 text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/10 via-transparent to-primary-500/10" />
            <div className="relative z-10">
              <h2 className="text-3xl font-bold text-white mb-4">
                Ready to Experience Smart Weather?
              </h2>
              <p className="text-white/60 mb-8 max-w-xl mx-auto">
                Join thousands of users who make better decisions every day with Mausam Vaani's intelligent forecasts.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link to="/signup">
                  <Button variant="primary" size="lg">
                    Get Started Free
                  </Button>
                </Link>
                <Link to="/demo">
                  <Button variant="secondary" size="lg">
                    View Live Demo
                  </Button>
                </Link>
              </div>
            </div>
          </GlassCard>
        </div>
      </section>
    </div>
  )
}

export default Features
