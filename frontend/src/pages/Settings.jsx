import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  User, MapPin, Globe, Bell, Smartphone, Moon, Palette,
  ChevronRight, Save, LogOut, Trash2, Shield
} from 'lucide-react'
import { GlassCard, Button, Input } from '../components/Shared'
import { useUser } from '../context/UserContext'

const Settings = () => {
  const { user, updateUser, logout } = useUser()
  const [activeTab, setActiveTab] = useState('profile')

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'preferences', label: 'Preferences', icon: Palette },
    { id: 'privacy', label: 'Privacy', icon: Shield },
  ]

  const languages = [
    { id: 'en', label: 'English' },
    { id: 'hi', label: 'हिन्दी (Hindi)' },
    { id: 'mr', label: 'मराठी (Marathi)' },
    { id: 'ta', label: 'தமிழ் (Tamil)' },
  ]

  const personas = [
    { id: 'farmer', label: 'Farmer', emoji: '👨‍🌾' },
    { id: 'commuter', label: 'Commuter', emoji: '🏢' },
    { id: 'logistics', label: 'Logistics', emoji: '🚚' },
    { id: 'homemaker', label: 'Homemaker', emoji: '🏠' },
  ]

  return (
    <div className="min-h-screen py-6">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-display font-bold text-white mb-2">Settings</h1>
          <p className="text-white/60">Manage your account and preferences</p>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <GlassCard className="p-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all text-left ${
                    activeTab === tab.id
                      ? 'bg-primary-500/20 text-primary-400'
                      : 'text-white/60 hover:bg-white/5 hover:text-white'
                  }`}
                >
                  <tab.icon className="w-5 h-5" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              ))}
              
              <div className="border-t border-white/10 mt-2 pt-2">
                <button
                  onClick={logout}
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-weather-storm hover:bg-weather-storm/10 transition-all text-left"
                >
                  <LogOut className="w-5 h-5" />
                  <span className="font-medium">Sign Out</span>
                </button>
              </div>
            </GlassCard>
          </div>

          {/* Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Profile Settings */}
            {activeTab === 'profile' && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-6"
              >
                <GlassCard className="p-6">
                  <h2 className="text-xl font-semibold text-white mb-6">Profile Information</h2>
                  
                  <div className="space-y-4">
                    <Input
                      label="Name"
                      icon={User}
                      defaultValue={user?.name || ''}
                      placeholder="Your name"
                    />
                    
                    <Input
                      label="Phone Number"
                      icon={Smartphone}
                      defaultValue={user?.phone || ''}
                      placeholder="+91 XXXXX XXXXX"
                    />
                    
                    <Input
                      label="Location"
                      icon={MapPin}
                      defaultValue={user?.location || 'Kothri Kalan, MP'}
                      placeholder="Your location"
                    />
                    
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-white/70">
                        <Globe className="inline w-4 h-4 mr-2" />
                        Language
                      </label>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {languages.map((lang) => (
                          <button
                            key={lang.id}
                            className={`p-3 rounded-xl text-sm font-medium transition-all ${
                              user?.language === lang.id
                                ? 'bg-primary-500/30 border-2 border-primary-500 text-primary-300'
                                : 'bg-white/5 border border-white/10 text-white/70 hover:bg-white/10'
                            }`}
                          >
                            {lang.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 pt-6 border-t border-white/10">
                    <Button variant="primary" icon={Save}>
                      Save Changes
                    </Button>
                  </div>
                </GlassCard>

                <GlassCard className="p-6">
                  <h2 className="text-xl font-semibold text-white mb-6">Your Persona</h2>
                  <p className="text-white/60 mb-4">
                    This helps us personalize your weather insights
                  </p>
                  
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {personas.map((persona) => (
                      <button
                        key={persona.id}
                        className={`p-4 rounded-xl text-center transition-all ${
                          user?.persona === persona.id
                            ? 'bg-primary-500/30 border-2 border-primary-500'
                            : 'bg-white/5 border border-white/10 hover:bg-white/10'
                        }`}
                      >
                        <span className="text-2xl mb-2 block">{persona.emoji}</span>
                        <span className={`text-sm ${user?.persona === persona.id ? 'text-white' : 'text-white/70'}`}>
                          {persona.label}
                        </span>
                      </button>
                    ))}
                  </div>
                </GlassCard>
              </motion.div>
            )}

            {/* Notifications Settings */}
            {activeTab === 'notifications' && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <GlassCard className="p-6">
                  <h2 className="text-xl font-semibold text-white mb-6">Notification Preferences</h2>
                  
                  <div className="space-y-4">
                    {[
                      { label: 'Push Notifications', desc: 'Receive alerts in browser' },
                      { label: 'SMS Alerts', desc: 'Get weather updates via SMS' },
                      { label: 'WhatsApp Alerts', desc: 'Daily forecasts on WhatsApp' },
                      { label: 'Severe Weather Alerts', desc: 'Immediate storm/flood warnings' },
                      { label: 'Crop Advisories', desc: 'Farming-specific notifications' },
                    ].map((item, index) => (
                      <div key={index} className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                        <div>
                          <p className="font-medium text-white">{item.label}</p>
                          <p className="text-sm text-white/50">{item.desc}</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" defaultChecked={index < 3} className="sr-only peer" />
                          <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
                        </label>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              </motion.div>
            )}

            {/* Preferences Settings */}
            {activeTab === 'preferences' && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <GlassCard className="p-6">
                  <h2 className="text-xl font-semibold text-white mb-6">App Preferences</h2>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                      <div>
                        <p className="font-medium text-white">Temperature Unit</p>
                        <p className="text-sm text-white/50">Choose Celsius or Fahrenheit</p>
                      </div>
                      <div className="flex gap-2">
                        <button className="px-4 py-2 rounded-lg bg-primary-500/20 text-primary-400 border border-primary-500/40">
                          °C
                        </button>
                        <button className="px-4 py-2 rounded-lg bg-white/5 text-white/60 border border-white/10">
                          °F
                        </button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                      <div>
                        <p className="font-medium text-white">Lite Mode</p>
                        <p className="text-sm text-white/50">Reduce animations for slow connections</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" className="sr-only peer" />
                        <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                      <div>
                        <p className="font-medium text-white">Dark Mode</p>
                        <p className="text-sm text-white/50">App theme preference</p>
                      </div>
                      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary-500/20 text-primary-400 border border-primary-500/40">
                        <Moon className="w-4 h-4" />
                        <span className="text-sm">Always Dark</span>
                      </div>
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            )}

            {/* Privacy Settings */}
            {activeTab === 'privacy' && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-6"
              >
                <GlassCard className="p-6">
                  <h2 className="text-xl font-semibold text-white mb-6">Privacy & Data</h2>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                      <div>
                        <p className="font-medium text-white">Location Sharing</p>
                        <p className="text-sm text-white/50">Allow access to your location for forecasts</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" defaultChecked className="sr-only peer" />
                        <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                      <div>
                        <p className="font-medium text-white">Analytics</p>
                        <p className="text-sm text-white/50">Help improve the app with usage data</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" defaultChecked className="sr-only peer" />
                        <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
                      </label>
                    </div>
                  </div>
                </GlassCard>

                <GlassCard className="p-6 border-weather-storm/30">
                  <h2 className="text-xl font-semibold text-weather-storm mb-4">Danger Zone</h2>
                  
                  <div className="space-y-3">
                    <button className="w-full flex items-center justify-between p-4 rounded-xl bg-weather-storm/10 text-weather-storm hover:bg-weather-storm/20 transition-all">
                      <div className="flex items-center gap-3">
                        <Trash2 className="w-5 h-5" />
                        <span>Delete Account</span>
                      </div>
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  </div>
                </GlassCard>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Settings
