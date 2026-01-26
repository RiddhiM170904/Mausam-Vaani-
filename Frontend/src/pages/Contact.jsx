import { useState } from 'react'
import { motion } from 'framer-motion'
import { Mail, MapPin, Send, MessageCircle, Clock, ChevronDown, ChevronUp } from 'lucide-react'
import { GlassCard, Button } from '../components/Shared'

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    subject: '',
    message: ''
  })
  const [expandedFaq, setExpandedFaq] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsSubmitting(true)
    await new Promise(resolve => setTimeout(resolve, 1000))
    alert('Thank you for your message! We will get back to you soon.')
    setFormData({ name: '', email: '', phone: '', subject: '', message: '' })
    setIsSubmitting(false)
  }

  const faqs = [
    {
      question: 'How accurate are your hyperlocal forecasts?',
      answer: 'Our forecasts achieve 95%+ accuracy by combining advanced AI models with multi-source data fusion and rigorous quality control processes.'
    },
    {
      question: 'Can I integrate Mausam Vaani into my application?',
      answer: 'Yes! We offer a comprehensive API ecosystem with unified JSON responses. Contact our sales team to learn more about our enterprise API packages.'
    },
    {
      question: 'Do you offer support in regional languages?',
      answer: 'Absolutely! Our AI-powered personalization layer supports multiple Indian languages including Hindi, Marathi, Tamil, and more.'
    },
    {
      question: 'What areas do you currently cover?',
      answer: 'We currently cover 500+ districts across India with street and village-scale resolution, and we are expanding coverage daily.'
    },
  ]

  return (
    <div className="min-h-screen py-12">
      {/* Hero Section */}
      <section className="px-4 sm:px-6 lg:px-8 mb-16">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="text-6xl mb-6 block">💬</span>
            <h1 className="text-4xl sm:text-5xl font-display font-bold text-white mb-6">
              Get in <span className="neon-text">Touch</span>
            </h1>
            <p className="text-xl text-white/60 max-w-3xl mx-auto">
              Have questions? We'd love to hear from you. Send us a message and we'll respond as soon as possible.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Contact Content */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-5 gap-12">
            {/* Contact Information */}
            <motion.div 
              className="lg:col-span-2 space-y-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div>
                <h2 className="text-2xl font-bold text-white mb-4">Contact Information</h2>
                <p className="text-white/60">
                  Reach out to us through any of these channels. Our team is here to help you.
                </p>
              </div>

              <div className="space-y-4">
                <GlassCard className="p-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-xl bg-primary-500/20 flex items-center justify-center flex-shrink-0 border border-primary-500/30">
                      <Mail className="w-5 h-5 text-primary-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white mb-1">Email</h3>
                      <p className="text-white/60 text-sm">info@mausamvaani.com</p>
                      <p className="text-white/60 text-sm">support@mausamvaani.com</p>
                    </div>
                  </div>
                </GlassCard>

                <GlassCard className="p-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center flex-shrink-0 border border-green-500/30">
                      <MessageCircle className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white mb-1">WhatsApp</h3>
                      <p className="text-white/60 text-sm">+91 123 456 7890</p>
                      <p className="text-white/60 text-sm">Quick responses within 1 hour</p>
                    </div>
                  </div>
                </GlassCard>

                <GlassCard className="p-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-xl bg-weather-sunny/20 flex items-center justify-center flex-shrink-0 border border-weather-sunny/30">
                      <Clock className="w-5 h-5 text-weather-sunny" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white mb-1">Business Hours</h3>
                      <div className="text-white/60 text-sm space-y-1">
                        <p>Mon - Fri: 9:00 AM - 6:00 PM</p>
                        <p>Saturday: 10:00 AM - 4:00 PM</p>
                        <p>Sunday: Closed</p>
                      </div>
                    </div>
                  </div>
                </GlassCard>

                <GlassCard className="p-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-xl bg-weather-rain/20 flex items-center justify-center flex-shrink-0 border border-weather-rain/30">
                      <MapPin className="w-5 h-5 text-weather-rain" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white mb-1">Office</h3>
                      <p className="text-white/60 text-sm">India</p>
                      <p className="text-white/60 text-sm">Physical office coming soon</p>
                    </div>
                  </div>
                </GlassCard>
              </div>
            </motion.div>

            {/* Contact Form */}
            <motion.div 
              className="lg:col-span-3"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <GlassCard className="p-8">
                <h2 className="text-2xl font-bold text-white mb-6">Send us a Message</h2>
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-white/70">Full Name *</label>
                      <input
                        type="text"
                        name="name"
                        required
                        value={formData.name}
                        onChange={handleChange}
                        className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none transition-all"
                        placeholder="Your name"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-white/70">Email Address *</label>
                      <input
                        type="email"
                        name="email"
                        required
                        value={formData.email}
                        onChange={handleChange}
                        className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none transition-all"
                        placeholder="your@email.com"
                      />
                    </div>
                  </div>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-white/70">Phone Number</label>
                      <input
                        type="tel"
                        name="phone"
                        value={formData.phone}
                        onChange={handleChange}
                        className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none transition-all"
                        placeholder="+91 98765 43210"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-white/70">Subject *</label>
                      <select
                        name="subject"
                        required
                        value={formData.subject}
                        onChange={handleChange}
                        className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none transition-all"
                      >
                        <option value="" className="bg-dark-800">Select a subject</option>
                        <option value="general" className="bg-dark-800">General Inquiry</option>
                        <option value="support" className="bg-dark-800">Technical Support</option>
                        <option value="sales" className="bg-dark-800">Sales</option>
                        <option value="partnership" className="bg-dark-800">Partnership</option>
                        <option value="feedback" className="bg-dark-800">Feedback</option>
                      </select>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-white/70">Message *</label>
                    <textarea
                      name="message"
                      required
                      value={formData.message}
                      onChange={handleChange}
                      rows="5"
                      className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none transition-all resize-none"
                      placeholder="Tell us more about your inquiry..."
                    />
                  </div>

                  <Button 
                    type="submit" 
                    variant="primary" 
                    size="lg" 
                    icon={Send}
                    loading={isSubmitting}
                    className="w-full"
                  >
                    Send Message
                  </Button>
                </form>
              </GlassCard>
            </motion.div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="px-4 sm:px-6 lg:px-8 mb-20">
        <div className="max-w-4xl mx-auto">
          <motion.h2 
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            className="text-3xl font-bold text-white mb-8 text-center"
          >
            Frequently Asked Questions
          </motion.h2>
          
          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlassCard className="overflow-hidden">
                  <button
                    onClick={() => setExpandedFaq(expandedFaq === index ? null : index)}
                    className="w-full p-6 flex items-center justify-between text-left"
                  >
                    <h3 className="font-semibold text-white pr-4">{faq.question}</h3>
                    {expandedFaq === index ? (
                      <ChevronUp className="w-5 h-5 text-primary-400 flex-shrink-0" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-white/40 flex-shrink-0" />
                    )}
                  </button>
                  {expandedFaq === index && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      className="px-6 pb-6"
                    >
                      <p className="text-white/60">{faq.answer}</p>
                    </motion.div>
                  )}
                </GlassCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

export default Contact
