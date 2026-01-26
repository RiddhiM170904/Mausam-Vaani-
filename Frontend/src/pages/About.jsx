import { motion } from 'framer-motion'
import { Target, Eye, Heart, Zap, Users, Shield } from 'lucide-react'
import { GlassCard, Button } from '../components/Shared'
import { Link } from 'react-router-dom'

const About = () => {
  const values = [
    { icon: Heart, title: 'Inclusivity', desc: 'Weather intelligence for everyone, regardless of technology access or language', color: 'text-pink-400' },
    { icon: Target, title: 'Accuracy', desc: 'Uncompromising commitment to precision through rigorous quality control', color: 'text-primary-400' },
    { icon: Zap, title: 'Innovation', desc: 'Continuously pushing boundaries with cutting-edge AI and data science', color: 'text-weather-sunny' },
    { icon: Users, title: 'Community', desc: 'Built with and for the communities we serve across India', color: 'text-green-400' },
  ]

  const stats = [
    { value: '1M+', label: 'Active Users' },
    { value: '500+', label: 'Districts Covered' },
    { value: '95%', label: 'Forecast Accuracy' },
    { value: '10+', label: 'Languages Supported' },
  ]

  return (
    <div className="min-h-screen py-12">
      {/* Hero Section */}
      <section className="px-4 mb-20 sm:px-6 lg:px-8">
        <div className="mx-auto text-center max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="block mb-6 text-6xl">🌤️</span>
            <h1 className="mb-6 text-4xl font-bold text-white sm:text-5xl lg:text-6xl font-display">
              About <span className="neon-text">Mausam Vaani</span>
            </h1>
            <p className="max-w-3xl mx-auto text-xl text-white/60">
              Transforming weather data into actionable intelligence for everyone, everywhere in India
            </p>
          </motion.div>
        </div>
      </section>

      {/* Mission & Vision */}
      <section className="px-4 mb-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="grid gap-8 md:grid-cols-2">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <GlassCard className="h-full p-8" accent="primary">
                <div className="flex items-center justify-center w-16 h-16 mb-6 border rounded-2xl bg-primary-500/20 border-primary-500/30">
                  <Target className="w-8 h-8 text-primary-400" />
                </div>
                <h2 className="mb-4 text-2xl font-bold text-white">Our Mission</h2>
                <p className="leading-relaxed text-white/70">
                  To democratize access to hyperlocal weather intelligence by combining cutting-edge AI, 
                  multi-source data fusion, and inclusive technology. We believe everyone deserves accurate, 
                  actionable weather insights regardless of their location or device.
                </p>
              </GlassCard>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <GlassCard className="h-full p-8" accent="sunny">
                <div className="flex items-center justify-center w-16 h-16 mb-6 border rounded-2xl bg-weather-sunny/20 border-weather-sunny/30">
                  <Eye className="w-8 h-8 text-weather-sunny" />
                </div>
                <h2 className="mb-4 text-2xl font-bold text-white">Our Vision</h2>
                <p className="leading-relaxed text-white/70">
                  To become the most trusted source of hyperlocal weather intelligence in India and beyond, 
                  empowering individuals, businesses, and governments to make informed decisions that save 
                  lives, protect livelihoods, and optimize operations.
                </p>
              </GlassCard>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Story Section */}
      <section className="px-4 mb-20 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <GlassCard className="relative p-8 overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 rounded-full bg-primary-500/10 blur-3xl" />
              <div className="absolute bottom-0 left-0 w-32 h-32 rounded-full bg-weather-sunny/10 blur-3xl" />
              
              <div className="relative z-10">
                <h2 className="mb-6 text-2xl font-bold text-center text-white">Our Story</h2>
                
                <div className="space-y-4 leading-relaxed text-white/70">
                  <p>
                    <span className="font-semibold text-primary-400">Mausam Vaani</span> was born from a simple observation: 
                    while weather forecasts are widely available, they often fail to provide the granular, actionable 
                    insights that people actually need. A farmer doesn't just need to know it might rain—they need 
                    to know if it will rain on their specific field, at what time, and whether they should irrigate or wait.
                  </p>
                  
                  <p>
                    Our founders, a team of AI researchers, meteorologists, and social entrepreneurs, came together 
                    with a vision to bridge this gap. By combining advanced AI models like 
                    <span className="font-medium text-weather-rain"> Temporal Fusion Transformers</span> with 
                    multi-source data fusion and natural language processing, we created a platform that doesn't 
                    just predict weather—<span className="font-medium text-white">it predicts impact</span>.
                  </p>
                  
                  <p>
                    Today, Mausam Vaani serves millions of users across India, from smallholder farmers in remote 
                    villages to large enterprises in metropolitan cities, helping them make better decisions through 
                    <span className="font-medium text-weather-sunny"> hyperlocal weather intelligence</span>.
                  </p>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        </div>
      </section>

      {/* Stats */}
      <section className="px-4 mb-20 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlassCard className="p-6 text-center">
                  <div className="mb-2 text-4xl font-bold neon-text">{stat.value}</div>
                  <div className="text-sm text-white/60">{stat.label}</div>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="px-4 mb-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <motion.h2 
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            className="mb-12 text-3xl font-bold text-center text-white"
          >
            Our Values
          </motion.h2>

          <div className="grid gap-6 md:grid-cols-4">
            {values.map((value, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlassCard className="h-full p-6 text-center">
                  <div className={`w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center bg-white/5`}>
                    <value.icon className={`w-6 h-6 ${value.color}`} />
                  </div>
                  <h3 className="mb-2 text-lg font-semibold text-white">{value.title}</h3>
                  <p className="text-sm text-white/60">{value.desc}</p>
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Team/Founder */}
   

      {/* CTA */}
      <section className="px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <GlassCard className="relative p-8 overflow-hidden text-center md:p-12">
            <div className="absolute inset-0 bg-gradient-to-r from-primary-500/10 via-transparent to-weather-sunny/10" />
            <div className="relative z-10">
              <h2 className="mb-4 text-3xl font-bold text-white">
                Join Our Journey
              </h2>
              <p className="max-w-xl mx-auto mb-8 text-white/60">
                Be part of the weather intelligence revolution in India. Start using Mausam Vaani today.
              </p>
              <div className="flex flex-col justify-center gap-4 sm:flex-row">
                <Link to="/signup">
                  <Button variant="primary" size="lg">
                    Get Started Free
                  </Button>
                </Link>
                <Link to="/contact">
                  <Button variant="secondary" size="lg">
                    Contact Us
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

export default About
