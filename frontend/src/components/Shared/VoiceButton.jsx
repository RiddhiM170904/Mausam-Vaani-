import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, MicOff, X } from 'lucide-react'

const VoiceButton = ({ onVoiceInput, className = '' }) => {
  const [isListening, setIsListening] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)

  const handleVoiceClick = () => {
    if (!isListening) {
      setIsListening(true)
      setIsExpanded(true)
      // Simulate voice recognition - in real app, use Web Speech API
      setTimeout(() => {
        setIsListening(false)
        if (onVoiceInput) {
          onVoiceInput("Sample voice input")
        }
      }, 3000)
    } else {
      setIsListening(false)
      setIsExpanded(false)
    }
  }

  return (
    <>
      {/* Floating Action Button */}
      <motion.button
        onClick={handleVoiceClick}
        className={`fixed bottom-6 right-6 z-50 w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 ${
          isListening 
            ? 'bg-weather-storm shadow-[0_0_30px_rgba(239,68,68,0.5)]' 
            : 'bg-primary-500 shadow-neon hover:shadow-neon-strong'
        } ${className}`}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: 'spring', stiffness: 260, damping: 20 }}
      >
        {isListening ? (
          <MicOff className="w-7 h-7 text-white" />
        ) : (
          <Mic className="w-7 h-7 text-white" />
        )}
        
        {/* Pulse Ring Animation */}
        {isListening && (
          <>
            <span className="absolute inset-0 rounded-full bg-weather-storm/50 animate-ping" />
            <span className="absolute inset-0 rounded-full bg-weather-storm/30 pulse-ring" />
          </>
        )}
      </motion.button>

      {/* Voice Modal */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-dark-900/80 backdrop-blur-md flex items-center justify-center p-4"
            onClick={() => {
              setIsExpanded(false)
              setIsListening(false)
            }}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="glass-card p-8 max-w-md w-full text-center space-y-6"
            >
              <button 
                onClick={() => {
                  setIsExpanded(false)
                  setIsListening(false)
                }}
                className="absolute top-4 right-4 text-white/50 hover:text-white transition-colors"
              >
                <X className="w-6 h-6" />
              </button>

              <div className="space-y-2">
                <h3 className="text-2xl font-display font-bold text-white">
                  मौसम सहायक
                </h3>
                <p className="text-white/60">Mausam Sahayak - Voice Assistant</p>
              </div>

              <div className="relative w-32 h-32 mx-auto">
                <div className={`absolute inset-0 rounded-full ${isListening ? 'bg-primary-500/20' : 'bg-white/10'} flex items-center justify-center`}>
                  <Mic className={`w-12 h-12 ${isListening ? 'text-primary-400' : 'text-white/50'}`} />
                </div>
                {isListening && (
                  <>
                    <span className="absolute inset-0 rounded-full border-2 border-primary-500/50 animate-ping" />
                    <span className="absolute inset-[-10px] rounded-full border border-primary-500/30 pulse-ring" style={{ animationDelay: '0.2s' }} />
                    <span className="absolute inset-[-20px] rounded-full border border-primary-500/20 pulse-ring" style={{ animationDelay: '0.4s' }} />
                  </>
                )}
              </div>

              <p className="text-white/70 text-lg">
                {isListening ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    <span className="ml-2">सुन रहा हूं...</span>
                  </span>
                ) : (
                  'माइक पर टैप करें और बोलें'
                )}
              </p>

              <div className="text-sm text-white/40">
                <p>Try: "कल बारिश होगी क्या?" or "What's the weather tomorrow?"</p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default VoiceButton
