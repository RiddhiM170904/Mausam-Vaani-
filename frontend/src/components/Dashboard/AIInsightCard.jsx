import { motion } from 'framer-motion'
import { 
  Lightbulb, Wheat, Car, Umbrella, Droplets, AlertTriangle,
  Share2, MessageCircle, ThumbsUp
} from 'lucide-react'
import { GlassCard } from '../Shared'

const AIInsightCard = ({
  insight = {
    type: 'farmer', // 'farmer' | 'commuter' | 'general' | 'alert'
    severity: 'info', // 'info' | 'warning' | 'danger'
    title: 'Crop Advisory',
    message: 'Humidity is high (85%). Risk of fungal blight. Do not irrigate today.',
    icon: Wheat,
    actionable: true,
    timestamp: 'Updated 10 min ago',
  },
  onShare,
  onLike,
}) => {
  const getSeverityStyles = () => {
    switch (insight.severity) {
      case 'warning':
        return {
          borderColor: 'border-l-weather-sunny',
          iconBg: 'bg-weather-sunny/20',
          iconColor: 'text-weather-sunny',
          accentColor: 'sunny',
        }
      case 'danger':
        return {
          borderColor: 'border-l-weather-storm',
          iconBg: 'bg-weather-storm/20',
          iconColor: 'text-weather-storm',
          accentColor: 'storm',
        }
      default:
        return {
          borderColor: 'border-l-primary-500',
          iconBg: 'bg-primary-500/20',
          iconColor: 'text-primary-400',
          accentColor: 'primary',
        }
    }
  }

  const getTypeIcon = () => {
    switch (insight.type) {
      case 'farmer':
        return Wheat
      case 'commuter':
        return Car
      case 'alert':
        return AlertTriangle
      default:
        return Lightbulb
    }
  }

  const styles = getSeverityStyles()
  const Icon = insight.icon || getTypeIcon()

  const handleWhatsAppShare = () => {
    const text = encodeURIComponent(`🌤️ Mausam Vaani Alert:\n\n${insight.title}\n${insight.message}\n\nDownload app: mausamvaani.com`)
    window.open(`https://wa.me/?text=${text}`, '_blank')
    if (onShare) onShare()
  }

  return (
    <GlassCard 
      className={`border-l-4 ${styles.borderColor} p-5`}
      accentColor={styles.accentColor}
    >
      <div className="flex items-start gap-4">
        {/* Icon */}
        <motion.div 
          className={`p-3 rounded-xl ${styles.iconBg} flex-shrink-0`}
          whileHover={{ scale: 1.1, rotate: 5 }}
        >
          <Icon className={`w-6 h-6 ${styles.iconColor}`} />
        </motion.div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-lg font-semibold text-white">{insight.title}</h3>
            {insight.severity === 'danger' && (
              <span className="px-2 py-0.5 text-xs font-medium bg-weather-storm/20 text-weather-storm rounded-full animate-pulse">
                URGENT
              </span>
            )}
          </div>
          
          <p className="text-white/70 leading-relaxed mb-3">{insight.message}</p>

          {/* Footer */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/40">{insight.timestamp}</span>
            
            <div className="flex items-center gap-2">
              {/* Like Button */}
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={onLike}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors group"
              >
                <ThumbsUp className="w-4 h-4 text-white/40 group-hover:text-primary-400" />
              </motion.button>

              {/* WhatsApp Share */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleWhatsAppShare}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-green-500/20 border border-green-500/30 text-green-400 hover:bg-green-500/30 transition-all text-sm font-medium"
              >
                <MessageCircle className="w-4 h-4" />
                <span className="hidden sm:inline">Share</span>
              </motion.button>
            </div>
          </div>
        </div>
      </div>
    </GlassCard>
  )
}

// Multiple insights wrapper
export const AIInsightsList = ({ insights = [], persona = 'farmer' }) => {
  const defaultInsights = {
    farmer: [
      {
        type: 'farmer',
        severity: 'warning',
        title: '🌾 Crop Advisory',
        message: 'Humidity is high (85%). Risk of fungal blight on wheat crop. Avoid irrigation today and consider fungicide spray.',
        timestamp: 'Updated 10 min ago',
      },
      {
        type: 'farmer',
        severity: 'info',
        title: '💧 Irrigation Tip',
        message: 'Light rain expected tomorrow (5mm). You can skip watering today to save water.',
        timestamp: 'Updated 30 min ago',
      },
    ],
    commuter: [
      {
        type: 'commuter',
        severity: 'warning',
        title: '🛵 Travel Alert',
        message: 'Heavy rain expected at 6 PM. Leave office by 5:30 PM to avoid traffic jams on main roads.',
        timestamp: 'Updated 15 min ago',
      },
      {
        type: 'commuter',
        severity: 'info',
        title: '☀️ Morning Commute',
        message: 'Clear weather expected until noon. Good conditions for bike/scooter.',
        timestamp: 'Updated 1 hour ago',
      },
    ],
    general: [
      {
        type: 'general',
        severity: 'info',
        title: '☂️ Weather Tip',
        message: 'Carry an umbrella today. 60% chance of evening showers.',
        timestamp: 'Updated 20 min ago',
      },
    ],
  }

  const displayInsights = insights.length > 0 ? insights : defaultInsights[persona] || defaultInsights.general

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-display font-bold text-white flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-primary-400" />
          AI Insights for You
        </h2>
        <span className="text-xs text-white/40">Personalized based on your profile</span>
      </div>
      
      {displayInsights.map((insight, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <AIInsightCard insight={insight} />
        </motion.div>
      ))}
    </div>
  )
}

export default AIInsightCard
