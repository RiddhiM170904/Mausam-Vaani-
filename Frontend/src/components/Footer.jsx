import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  CloudRain, Mail, Phone, MapPin, 
  Facebook, Twitter, Instagram, Linkedin, Youtube,
  MessageCircle, Send
} from 'lucide-react'

const Footer = () => {
  const currentYear = new Date().getFullYear()

  const footerLinks = {
    product: [
      { name: 'Features', path: '/features' },
      { name: 'Dashboard', path: '/dashboard' },
      { name: 'Planner', path: '/planner' },
      { name: 'Maps', path: '/maps' },
    ],
    company: [
      { name: 'About Us', path: '/about' },
      { name: 'Contact', path: '/contact' },
      { name: 'Careers', path: '/careers' },
    ],
    legal: [
      { name: 'Privacy Policy', path: '/privacy' },
      { name: 'Terms of Service', path: '/terms' },
    ],
  }

  const socialLinks = [
    { icon: Facebook, href: '#', label: 'Facebook' },
    { icon: Twitter, href: '#', label: 'Twitter' },
    { icon: Instagram, href: '#', label: 'Instagram' },
    { icon: Linkedin, href: '#', label: 'LinkedIn' },
    { icon: Youtube, href: '#', label: 'YouTube' },
  ]

  return (
    <footer className="relative bg-dark-900 border-t border-white/10">
      <div className="absolute inset-0 bg-gradient-to-t from-dark-900 via-dark-900/95 to-transparent pointer-events-none" />
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-12 mb-12">
          {/* Brand Section */}
          <div className="lg:col-span-2 space-y-6">
            <Link to="/" className="flex items-center gap-2">
              <CloudRain className="h-8 w-8 text-primary-400 icon-glow-primary" />
              <span className="text-2xl font-display font-bold text-white">
                Mausam <span className="text-primary-400">Vaani</span>
              </span>
            </Link>
            <p className="text-white/60 leading-relaxed max-w-sm">
              Hyperlocal weather intelligence powered by AI. Empowering farmers, commuters, and businesses with precise forecasts and actionable insights.
            </p>

            {/* Newsletter */}
            <div className="space-y-3">
              <p className="text-sm font-medium text-white">Get weather alerts directly:</p>
              <div className="flex gap-2">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/40 focus:outline-none focus:border-primary-500/50"
                />
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-4 py-3 bg-primary-500 rounded-xl text-white hover:bg-primary-600 shadow-neon transition-all"
                >
                  <Send className="w-5 h-5" />
                </motion.button>
              </div>
            </div>

            {/* WhatsApp CTA */}
            <motion.a
              href="https://wa.me/911234567890"
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.02 }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-500/40 rounded-xl text-green-400 hover:bg-green-500/30 transition-all"
            >
              <MessageCircle className="w-5 h-5" />
              <span className="text-sm font-medium">Get alerts on WhatsApp</span>
            </motion.a>
          </div>

          {/* Links Grid */}
          <div className="lg:col-span-3 grid grid-cols-2 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-4">Product</h3>
              <ul className="space-y-3">
                {footerLinks.product.map((link) => (
                  <li key={link.name}>
                    <Link to={link.path} className="text-white/60 hover:text-primary-400 transition-colors text-sm">
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-4">Company</h3>
              <ul className="space-y-3">
                {footerLinks.company.map((link) => (
                  <li key={link.name}>
                    <Link to={link.path} className="text-white/60 hover:text-primary-400 transition-colors text-sm">
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-4">Contact</h3>
              <ul className="space-y-3">
                <li className="flex items-center gap-2 text-white/60 text-sm">
                  <Mail className="w-4 h-4 text-primary-400" />
                  <span>info@mausamvaani.com</span>
                </li>
                <li className="flex items-center gap-2 text-white/60 text-sm">
                  <Phone className="w-4 h-4 text-primary-400" />
                  <span>+91 123 456 7890</span>
                </li>
                <li className="flex items-center gap-2 text-white/60 text-sm">
                  <MapPin className="w-4 h-4 text-primary-400" />
                  <span>India</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-white/40">
            © {currentYear} Mausam Vaani. All rights reserved. Made with ❤️ in India
          </p>

          <div className="flex items-center gap-6">
            {footerLinks.legal.map((link) => (
              <Link key={link.name} to={link.path} className="text-sm text-white/40 hover:text-white/70 transition-colors">
                {link.name}
              </Link>
            ))}
          </div>

          <div className="flex items-center gap-4">
            {socialLinks.map((social) => (
              <motion.a
                key={social.label}
                href={social.href}
                aria-label={social.label}
                whileHover={{ scale: 1.2, y: -2 }}
                className="p-2 rounded-lg text-white/40 hover:text-primary-400 hover:bg-white/5 transition-all"
              >
                <social.icon className="w-5 h-5" />
              </motion.a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
