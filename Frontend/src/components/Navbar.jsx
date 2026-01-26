import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Menu, X, CloudRain, Search, Bell, User, MapPin, 
  Mic, Settings, LogOut, ChevronDown 
} from 'lucide-react'

const Navbar = ({ isAuthenticated = false, user = null }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [showSearch, setShowSearch] = useState(false)
  const [showProfileMenu, setShowProfileMenu] = useState(false)
  const [hasNotifications, setHasNotifications] = useState(true)
  const location = useLocation()
  const navigate = useNavigate()

  const toggleMenu = () => setIsOpen(!isOpen)

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Dashboard', path: '/dashboard' },
    { name: 'Planner', path: '/planner' },
    { name: 'Maps', path: '/maps' },
    { name: 'Community', path: '/community' },
  ]

  const publicLinks = [
    { name: 'Home', path: '/' },
    { name: 'Features', path: '/features' },
    { name: 'About', path: '/about' },
    { name: 'Contact', path: '/contact' },
  ]

  const links = isAuthenticated ? navLinks : publicLinks

  const isActive = (path) => location.pathname === path

  return (
    <nav className="sticky top-0 z-50 bg-dark-800/80 backdrop-blur-xl border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo & Location */}
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center gap-2">
              <div className="relative">
                <CloudRain className="h-8 w-8 text-primary-400 icon-glow-primary" />
                <span className="absolute -top-1 -right-1 w-2 h-2 bg-weather-rain rounded-full animate-pulse" />
              </div>
              <span className="text-xl font-display font-bold text-white hidden sm:block">
                Mausam <span className="text-primary-400">Vaani</span>
              </span>
            </Link>

            {/* Current Location (Desktop) */}
            {isAuthenticated && (
              <div className="hidden lg:flex items-center gap-1 text-white/60 text-sm">
                <MapPin className="w-4 h-4 text-primary-400" />
                <span>{user?.location || 'Kothri Kalan, MP'}</span>
              </div>
            )}
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {links.map((link) => (
              <Link
                key={link.name}
                to={link.path}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                  isActive(link.path)
                    ? 'bg-primary-500/20 text-primary-400 neon-border'
                    : 'text-white/70 hover:text-white hover:bg-white/10'
                }`}
              >
                {link.name}
              </Link>
            ))}
          </div>

          {/* Right Side Actions */}
          <div className="flex items-center gap-3">
            {/* Search Bar (Desktop) */}
            <AnimatePresence>
              {showSearch && (
                <motion.div
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: 250, opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  className="hidden md:block overflow-hidden"
                >
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search location..."
                      className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-10 py-2 text-sm text-white placeholder-white/40 focus:outline-none focus:border-primary-500/50"
                      autoFocus
                    />
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
                    <button className="absolute right-3 top-1/2 -translate-y-1/2">
                      <Mic className="w-4 h-4 text-primary-400 hover:text-primary-300 transition-colors" />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <button
              onClick={() => setShowSearch(!showSearch)}
              className="hidden md:flex p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-all duration-300"
            >
              <Search className="w-5 h-5" />
            </button>

            {/* Notification Bell */}
            {isAuthenticated && (
              <button className="relative p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-all duration-300">
                <Bell className="w-5 h-5" />
                {hasNotifications && (
                  <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-weather-storm rounded-full border-2 border-dark-800" />
                )}
              </button>
            )}

            {/* Profile / Auth */}
            {isAuthenticated ? (
              <div className="relative">
                <button
                  onClick={() => setShowProfileMenu(!showProfileMenu)}
                  className="flex items-center gap-2 p-1.5 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-300"
                >
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                  </div>
                  <ChevronDown className={`w-4 h-4 text-white/60 transition-transform hidden sm:block ${showProfileMenu ? 'rotate-180' : ''}`} />
                </button>

                {/* Profile Dropdown */}
                <AnimatePresence>
                  {showProfileMenu && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}
                      className="absolute right-0 mt-2 w-56 glass-card p-2 space-y-1"
                    >
                      <div className="px-3 py-2 border-b border-white/10">
                        <p className="font-medium text-white">{user?.name || 'User'}</p>
                        <p className="text-sm text-white/50">{user?.persona || 'Farmer'}</p>
                      </div>
                      <Link
                        to="/settings"
                        onClick={() => setShowProfileMenu(false)}
                        className="flex items-center gap-3 px-3 py-2 rounded-lg text-white/70 hover:bg-white/10 hover:text-white transition-all"
                      >
                        <Settings className="w-4 h-4" />
                        Settings
                      </Link>
                      <button
                        onClick={() => {
                          setShowProfileMenu(false)
                          // Handle logout
                        }}
                        className="flex items-center gap-3 w-full px-3 py-2 rounded-lg text-weather-storm hover:bg-weather-storm/10 transition-all"
                      >
                        <LogOut className="w-4 h-4" />
                        Sign Out
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Link
                  to="/login"
                  className="px-4 py-2 rounded-lg text-white/70 hover:text-white hover:bg-white/10 transition-all duration-300 text-sm font-medium"
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className="px-4 py-2 rounded-xl bg-primary-500 text-white hover:bg-primary-600 shadow-neon hover:shadow-neon-strong transition-all duration-300 text-sm font-medium"
                >
                  Get Started
                </Link>
              </div>
            )}

            {/* Mobile menu button */}
            <button
              onClick={toggleMenu}
              className="md:hidden p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-all"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-dark-800/95 backdrop-blur-xl border-t border-white/10"
          >
            <div className="px-4 py-4 space-y-2">
              {/* Mobile Search */}
              <div className="relative mb-4">
                <input
                  type="text"
                  placeholder="Search location..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-10 py-3 text-sm text-white placeholder-white/40 focus:outline-none focus:border-primary-500/50"
                />
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
                <button className="absolute right-3 top-1/2 -translate-y-1/2">
                  <Mic className="w-5 h-5 text-primary-400" />
                </button>
              </div>

              {links.map((link) => (
                <Link
                  key={link.name}
                  to={link.path}
                  onClick={() => setIsOpen(false)}
                  className={`block px-4 py-3 rounded-xl text-base font-medium transition-all ${
                    isActive(link.path)
                      ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30'
                      : 'text-white/70 hover:text-white hover:bg-white/10'
                  }`}
                >
                  {link.name}
                </Link>
              ))}

              {!isAuthenticated && (
                <div className="pt-4 space-y-2 border-t border-white/10 mt-4">
                  <Link
                    to="/login"
                    onClick={() => setIsOpen(false)}
                    className="block w-full px-4 py-3 rounded-xl text-center font-medium text-white/70 hover:text-white hover:bg-white/10 transition-all"
                  >
                    Login
                  </Link>
                  <Link
                    to="/signup"
                    onClick={() => setIsOpen(false)}
                    className="block w-full px-4 py-3 rounded-xl text-center font-medium bg-primary-500 text-white hover:bg-primary-600 shadow-neon transition-all"
                  >
                    Get Started
                  </Link>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  )
}

export default Navbar
