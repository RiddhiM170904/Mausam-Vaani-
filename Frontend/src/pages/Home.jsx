import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  MapPin, Sparkles, TrendingUp, Users, Smartphone, Zap, 
  Cloud, CloudRain, Sun, ArrowRight, Play, ChevronRight,
  Wheat, Car, MessageCircle, Globe
} from 'lucide-react'
import { GlassCard, Button } from '../components/Shared'

const Home = () => {
  const features = [
    {
      icon: MapPin,
      title: 'Hyperlocal Forecasting',
      description: 'Street & village-scale micro-climate resolution powered by AI',
      color: 'primary',
    },
    {
      icon: Sparkles,
      title: 'Data Fusion',
      description: 'IMD feeds + private APIs + IoT + crowdsourcing combined',
      color: 'weather-rain',
    },
    {
      icon: TrendingUp,
      title: 'Impact Intelligence',
      description: 'Weather → Activity / Crop / Sector actionable outcomes',
      color: 'weather-sunny',
    },
    {
      icon: Users,
      title: 'Personalization',
      description: 'Gemini LLM powered multi-role, multi-language advisories',
      color: 'green',
    },
    {
      icon: Smartphone,
      title: 'Accessibility',
      description: 'Web + SMS + WhatsApp + Voice for inclusive reach',
      color: 'purple',
    },
    {
      icon: Zap,
      title: 'API Ecosystem',
      description: 'Enterprise & government integration ready',
      color: 'orange',
    },
  ]

  const personas = [
    { icon: Wheat, title: 'Farmers', desc: 'Crop-specific advisories', color: 'weather-sunny' },
    { icon: Car, title: 'Commuters', desc: 'Travel & route planning', color: 'weather-rain' },
    { icon: Globe, title: 'Businesses', desc: 'Industry insights', color: 'primary' },
  ]

  const stats = [
    { value: '99.9%', label: 'Accuracy' },
    { value: '10M+', label: 'Users Served' },
    { value: '500+', label: 'Cities Covered' },
    { value: '24/7', label: 'Live Updates' },
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 animated-gradient" />
        
        {/* Floating Elements */}
        <div className="absolute inset-0 overflow-hidden">
          {/* Glowing orbs */}
          <motion.div
            className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl"
            animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 8, repeat: Infinity }}
          />
          <motion.div
            className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-weather-rain/10 rounded-full blur-3xl"
            animate={{ scale: [1.2, 1, 1.2], opacity: [0.4, 0.2, 0.4] }}
            transition={{ duration: 6, repeat: Infinity }}
          />
          
          {/* Floating clouds */}
          <motion.div
            className="absolute top-20 left-10"
            animate={{ x: [0, 100, 0], y: [0, -20, 0] }}
            transition={{ duration: 20, repeat: Infinity }}
          >
            <Cloud className="w-20 h-20 text-white/5" />
          </motion.div>
          <motion.div
            className="absolute bottom-40 right-20"
            animate={{ x: [0, -80, 0], y: [0, 15, 0] }}
            transition={{ duration: 15, repeat: Infinity }}
          >
            <CloudRain className="w-16 h-16 text-weather-rain/10" />
          </motion.div>
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500/10 border border-primary-500/30 mb-8"
            >
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-sm text-primary-300">Powered by AI & TFT Models</span>
            </motion.div>

            <h1 className="text-4xl sm:text-5xl lg:text-7xl font-display font-bold text-white mb-6 leading-tight">
              Hyperlocal Weather
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 via-weather-rain to-weather-sunny">
                Intelligence for India
              </span>
            </h1>

            <p className="text-lg sm:text-xl text-white/60 max-w-3xl mx-auto mb-10 leading-relaxed">
              From village-scale forecasts to personalized crop advisories. 
              Empowering farmers, commuters, and businesses with actionable weather insights.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/signup">
                <Button variant="primary" size="lg" icon={ArrowRight} iconPosition="right">
                  Get Started Free
                </Button>
              </Link>
              <Link to="/demo">
                <Button variant="secondary" size="lg" icon={Play}>
                  Watch Demo
                </Button>
              </Link>
            </div>
          </motion.div>

          {/* Hero Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4"
          >
            {stats.map((stat, index) => (
              <GlassCard key={index} className="p-4 text-center" hover>
                <p className="text-2xl sm:text-3xl font-bold text-white neon-text">{stat.value}</p>
                <p className="text-sm text-white/50">{stat.label}</p>
              </GlassCard>
            ))}
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-2">
            <motion.div
              className="w-1.5 h-1.5 bg-primary-400 rounded-full"
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </div>
        </motion.div>
      </section>

      {/* Personas Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 relative">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl sm:text-4xl font-display font-bold text-white mb-4">
              Built for Everyone
            </h2>
            <p className="text-white/60 max-w-2xl mx-auto">
              Personalized weather intelligence tailored to your specific needs
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {personas.map((persona, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <GlassCard className="p-8 text-center" hover>
                  <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl bg-${persona.color}/20 flex items-center justify-center`}>
                    <persona.icon className={`w-8 h-8 text-${persona.color === 'primary' ? 'primary-400' : persona.color}`} />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">{persona.title}</h3>
                  <p className="text-white/60">{persona.desc}</p>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-display font-bold text-white mb-4">
              Core Value Proposition
            </h2>
            <p className="text-white/60 max-w-2xl mx-auto">
              Six layers of innovation delivering weather intelligence that matters
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <GlassCard className="p-6 h-full" hover>
                  <div className={`w-12 h-12 rounded-xl bg-${feature.color === 'primary' ? 'primary-500' : feature.color}/20 flex items-center justify-center mb-4`}>
                    <feature.icon className={`w-6 h-6 ${feature.color === 'primary' ? 'text-primary-400' : `text-${feature.color}`}`} />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                  <p className="text-white/60 text-sm leading-relaxed">{feature.description}</p>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* WhatsApp CTA */}
      <section className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <GlassCard className="p-8 sm:p-12 text-center relative overflow-hidden" accentColor="primary">
            <div className="absolute inset-0 bg-gradient-to-r from-green-500/5 to-primary-500/5" />
            <div className="relative z-10">
              <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-green-500/20 flex items-center justify-center">
                <MessageCircle className="w-8 h-8 text-green-400" />
              </div>
              <h2 className="text-2xl sm:text-3xl font-display font-bold text-white mb-4">
                Get Weather Alerts on WhatsApp
              </h2>
              <p className="text-white/60 mb-8 max-w-xl mx-auto">
                Receive daily forecasts, crop advisories, and severe weather alerts directly on WhatsApp. Perfect for areas with limited internet.
              </p>
              <motion.a
                href="https://wa.me/911234567890"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-green-500 text-white font-semibold hover:bg-green-600 transition-all shadow-lg"
              >
                <MessageCircle className="w-5 h-5" />
                Connect on WhatsApp
                <ChevronRight className="w-5 h-5" />
              </motion.a>
            </div>
          </GlassCard>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl sm:text-4xl font-display font-bold text-white mb-6">
              Ready to Experience Hyperlocal Weather Intelligence?
            </h2>
            <p className="text-white/60 mb-8 text-lg">
              Join thousands of users who trust Mausam Vaani for precise, actionable weather insights
            </p>
            <Link to="/signup">
              <Button variant="primary" size="lg" icon={ArrowRight} iconPosition="right">
                Start Your Free Trial
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default Home
