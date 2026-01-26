import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { 
  User, Phone, Globe, MapPin, Navigation, 
  Tractor, Briefcase, Truck, Home, HardHat,
  Wheat, Droplets, CloudRain, Bike, Car, Train,
  ChevronRight, ChevronLeft, Check, Loader2
} from 'lucide-react'
import { GlassCard, Button, Input } from '../Shared'

const PERSONAS = [
  { id: 'farmer', icon: Tractor, label: 'Farmer', labelHi: 'किसान', color: 'weather-sunny' },
  { id: 'commuter', icon: Briefcase, label: 'Office Goer', labelHi: 'कार्यालय कर्मी', color: 'primary' },
  { id: 'logistics', icon: Truck, label: 'Logistics/Driver', labelHi: 'ड्राइवर', color: 'weather-rain' },
  { id: 'homemaker', icon: Home, label: 'Homemaker', labelHi: 'गृहिणी', color: 'green' },
  { id: 'construction', icon: HardHat, label: 'Construction', labelHi: 'निर्माण', color: 'orange' },
]

const CROPS = [
  { id: 'wheat', label: 'Wheat (गेहूं)', icon: '🌾' },
  { id: 'rice', label: 'Rice (चावल)', icon: '🍚' },
  { id: 'cotton', label: 'Cotton (कपास)', icon: '🧶' },
  { id: 'sugarcane', label: 'Sugarcane (गन्ना)', icon: '🎋' },
  { id: 'vegetables', label: 'Vegetables (सब्जियां)', icon: '🥬' },
  { id: 'pulses', label: 'Pulses (दालें)', icon: '🫘' },
]

const TRANSPORT_MODES = [
  { id: 'bike', icon: Bike, label: 'Bike/Scooter' },
  { id: 'car', icon: Car, label: 'Car' },
  { id: 'metro', icon: Train, label: 'Metro/Train' },
  { id: 'walk', icon: User, label: 'Walking' },
]

const LANGUAGES = [
  { id: 'en', label: 'English' },
  { id: 'hi', label: 'हिन्दी (Hindi)' },
  { id: 'mr', label: 'मराठी (Marathi)' },
  { id: 'ta', label: 'தமிழ் (Tamil)' },
  { id: 'te', label: 'తెలుగు (Telugu)' },
  { id: 'bn', label: 'বাংলা (Bengali)' },
]

const OnboardingWizard = () => {
  const navigate = useNavigate()
  const [currentStep, setCurrentStep] = useState(1)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isLocating, setIsLocating] = useState(false)
  
  const [formData, setFormData] = useState({
    name: '',
    phone: '',
    language: 'en',
    location: '',
    pincode: '',
    persona: '',
    // Farmer specific
    crops: [],
    soilType: '',
    irrigationType: '',
    // Commuter specific
    transportMode: '',
  })

  const totalSteps = 4

  const updateFormData = (key, value) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }

  const handleAutoDetectLocation = () => {
    setIsLocating(true)
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          // In real app, reverse geocode this
          updateFormData('location', 'Kothri Kalan, MP')
          setIsLocating(false)
        },
        (error) => {
          console.error(error)
          setIsLocating(false)
        }
      )
    } else {
      setIsLocating(false)
    }
  }

  const handleNext = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleSubmit = async () => {
    setIsSubmitting(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsSubmitting(false)
    navigate('/dashboard')
  }

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return formData.name && formData.phone && formData.language
      case 2:
        return formData.location || formData.pincode
      case 3:
        return formData.persona
      case 4:
        if (formData.persona === 'farmer') {
          return formData.crops.length > 0
        }
        if (formData.persona === 'commuter' || formData.persona === 'logistics') {
          return formData.transportMode
        }
        return true
      default:
        return true
    }
  }

  const slideVariants = {
    enter: (direction) => ({
      x: direction > 0 ? 300 : -300,
      opacity: 0,
    }),
    center: {
      x: 0,
      opacity: 1,
    },
    exit: (direction) => ({
      x: direction < 0 ? 300 : -300,
      opacity: 0,
    }),
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <GlassCard className="w-full max-w-2xl p-8" animate={false}>
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-white/60">Step {currentStep} of {totalSteps}</span>
            <span className="text-sm text-primary-400">{Math.round((currentStep / totalSteps) * 100)}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-primary-500 to-weather-rain"
              initial={{ width: 0 }}
              animate={{ width: `${(currentStep / totalSteps) * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        {/* Step Content */}
        <AnimatePresence mode="wait" custom={currentStep}>
          <motion.div
            key={currentStep}
            custom={currentStep}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.3 }}
            className="min-h-[350px]"
          >
            {/* Step 1: Basic Info */}
            {currentStep === 1 && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-display font-bold text-white mb-2">
                    Welcome to Mausam Vaani
                  </h2>
                  <p className="text-white/60">Let's personalize your weather experience</p>
                </div>

                <Input
                  label="Your Name"
                  icon={User}
                  placeholder="Enter your name"
                  value={formData.name}
                  onChange={(e) => updateFormData('name', e.target.value)}
                />

                <Input
                  label="Mobile Number"
                  icon={Phone}
                  placeholder="+91 XXXXX XXXXX"
                  value={formData.phone}
                  onChange={(e) => updateFormData('phone', e.target.value)}
                />

                <div className="space-y-2">
                  <label className="block text-sm font-medium text-white/70">
                    <Globe className="inline w-4 h-4 mr-2" />
                    Preferred Language
                  </label>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {LANGUAGES.map((lang) => (
                      <button
                        key={lang.id}
                        onClick={() => updateFormData('language', lang.id)}
                        className={`p-3 rounded-xl text-sm font-medium transition-all ${
                          formData.language === lang.id
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
            )}

            {/* Step 2: Location */}
            {currentStep === 2 && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-display font-bold text-white mb-2">
                    Your Location
                  </h2>
                  <p className="text-white/60">We need this for hyperlocal forecasts</p>
                </div>

                <motion.button
                  onClick={handleAutoDetectLocation}
                  disabled={isLocating}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full p-6 rounded-2xl bg-gradient-to-r from-primary-500/20 to-weather-rain/20 border-2 border-dashed border-primary-500/50 text-center transition-all hover:border-primary-500"
                >
                  <div className="relative inline-block">
                    <Navigation className={`w-12 h-12 text-primary-400 mx-auto mb-3 ${isLocating ? 'animate-pulse' : ''}`} />
                    {isLocating && (
                      <span className="absolute inset-0 rounded-full border-2 border-primary-500/50 animate-ping" />
                    )}
                  </div>
                  <p className="text-lg font-medium text-white">
                    {isLocating ? 'Detecting...' : 'Auto-Detect My Location'}
                  </p>
                  <p className="text-sm text-white/50 mt-1">Uses GPS for precise location</p>
                </motion.button>

                {formData.location && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 rounded-xl bg-green-500/10 border border-green-500/30 flex items-center gap-3"
                  >
                    <Check className="w-5 h-5 text-green-400" />
                    <span className="text-green-300">{formData.location}</span>
                  </motion.div>
                )}

                <div className="flex items-center gap-4">
                  <div className="flex-1 h-px bg-white/10" />
                  <span className="text-white/40 text-sm">OR</span>
                  <div className="flex-1 h-px bg-white/10" />
                </div>

                <Input
                  label="Enter Pincode Manually"
                  icon={MapPin}
                  placeholder="Enter 6-digit pincode"
                  value={formData.pincode}
                  onChange={(e) => updateFormData('pincode', e.target.value)}
                />
              </div>
            )}

            {/* Step 3: Persona Selection */}
            {currentStep === 3 && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-display font-bold text-white mb-2">
                    Who Are You?
                  </h2>
                  <p className="text-white/60">This helps us personalize your insights</p>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {PERSONAS.map((persona) => (
                    <motion.button
                      key={persona.id}
                      onClick={() => updateFormData('persona', persona.id)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className={`p-6 rounded-2xl text-center transition-all ${
                        formData.persona === persona.id
                          ? 'bg-primary-500/30 border-2 border-primary-500 shadow-neon'
                          : 'bg-white/5 border border-white/10 hover:bg-white/10'
                      }`}
                    >
                      <persona.icon className={`w-10 h-10 mx-auto mb-3 ${
                        formData.persona === persona.id ? 'text-primary-400' : 'text-white/60'
                      }`} />
                      <p className={`font-medium ${
                        formData.persona === persona.id ? 'text-white' : 'text-white/70'
                      }`}>
                        {persona.label}
                      </p>
                      <p className="text-sm text-white/40">{persona.labelHi}</p>
                    </motion.button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 4: Deep Dive */}
            {currentStep === 4 && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-display font-bold text-white mb-2">
                    {formData.persona === 'farmer' ? 'Your Farm Details' : 'Your Preferences'}
                  </h2>
                  <p className="text-white/60">
                    {formData.persona === 'farmer' 
                      ? 'Help us give you crop-specific advisories'
                      : 'Customize your weather alerts'
                    }
                  </p>
                </div>

                {/* Farmer Deep Dive */}
                {formData.persona === 'farmer' && (
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-white/70">
                        Select Your Crops (choose all that apply)
                      </label>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        {CROPS.map((crop) => (
                          <button
                            key={crop.id}
                            onClick={() => {
                              const crops = formData.crops.includes(crop.id)
                                ? formData.crops.filter(c => c !== crop.id)
                                : [...formData.crops, crop.id]
                              updateFormData('crops', crops)
                            }}
                            className={`p-3 rounded-xl text-left transition-all flex items-center gap-2 ${
                              formData.crops.includes(crop.id)
                                ? 'bg-weather-sunny/20 border-2 border-weather-sunny/50 text-weather-sunny'
                                : 'bg-white/5 border border-white/10 text-white/70 hover:bg-white/10'
                            }`}
                          >
                            <span className="text-xl">{crop.icon}</span>
                            <span className="text-sm font-medium">{crop.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="block text-sm font-medium text-white/70">Soil Type</label>
                        <select
                          value={formData.soilType}
                          onChange={(e) => updateFormData('soilType', e.target.value)}
                          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary-500/50"
                        >
                          <option value="" className="bg-dark-800">Select...</option>
                          <option value="alluvial" className="bg-dark-800">Alluvial</option>
                          <option value="black" className="bg-dark-800">Black</option>
                          <option value="red" className="bg-dark-800">Red</option>
                          <option value="laterite" className="bg-dark-800">Laterite</option>
                        </select>
                      </div>

                      <div className="space-y-2">
                        <label className="block text-sm font-medium text-white/70">Irrigation</label>
                        <select
                          value={formData.irrigationType}
                          onChange={(e) => updateFormData('irrigationType', e.target.value)}
                          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary-500/50"
                        >
                          <option value="" className="bg-dark-800">Select...</option>
                          <option value="canal" className="bg-dark-800">Canal</option>
                          <option value="well" className="bg-dark-800">Well/Borewell</option>
                          <option value="rainfed" className="bg-dark-800">Rain-fed</option>
                          <option value="drip" className="bg-dark-800">Drip/Sprinkler</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}

                {/* Commuter/Logistics Deep Dive */}
                {(formData.persona === 'commuter' || formData.persona === 'logistics') && (
                  <div className="space-y-4">
                    <label className="block text-sm font-medium text-white/70">
                      Your Primary Transport Mode
                    </label>
                    <div className="grid grid-cols-2 gap-4">
                      {TRANSPORT_MODES.map((mode) => (
                        <motion.button
                          key={mode.id}
                          onClick={() => updateFormData('transportMode', mode.id)}
                          whileHover={{ scale: 1.03 }}
                          whileTap={{ scale: 0.97 }}
                          className={`p-4 rounded-xl flex items-center gap-3 transition-all ${
                            formData.transportMode === mode.id
                              ? 'bg-weather-rain/20 border-2 border-weather-rain/50'
                              : 'bg-white/5 border border-white/10 hover:bg-white/10'
                          }`}
                        >
                          <mode.icon className={`w-6 h-6 ${
                            formData.transportMode === mode.id ? 'text-weather-rain' : 'text-white/60'
                          }`} />
                          <span className={formData.transportMode === mode.id ? 'text-white' : 'text-white/70'}>
                            {mode.label}
                          </span>
                        </motion.button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Other Personas */}
                {!['farmer', 'commuter', 'logistics'].includes(formData.persona) && (
                  <div className="text-center py-8">
                    <Check className="w-16 h-16 text-green-400 mx-auto mb-4" />
                    <p className="text-white/70">
                      Great! We have all we need to personalize your experience.
                    </p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Navigation Buttons */}
        <div className="flex items-center justify-between mt-8 pt-6 border-t border-white/10">
          <Button
            variant="ghost"
            onClick={handleBack}
            disabled={currentStep === 1}
            icon={ChevronLeft}
          >
            Back
          </Button>

          {currentStep < totalSteps ? (
            <Button
              variant="primary"
              onClick={handleNext}
              disabled={!canProceed()}
              icon={ChevronRight}
              iconPosition="right"
            >
              Continue
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={handleSubmit}
              disabled={!canProceed()}
              loading={isSubmitting}
            >
              {isSubmitting ? 'Setting up...' : 'Start Exploring'}
            </Button>
          )}
        </div>
      </GlassCard>
    </div>
  )
}

export default OnboardingWizard
