import { motion } from 'framer-motion'
import { Sparkles, AlertTriangle, Info, CheckCircle } from 'lucide-react'
import AIInsightCard from './AIInsightCard'

const AIInsightsList = ({ insights = [], persona }) => {
  if (!insights || insights.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6"
      >
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="w-5 h-5 text-primary-400" />
          <h2 className="text-lg font-semibold text-white">AI Insights</h2>
        </div>
        <p className="text-white/50 text-center py-8">
          Loading personalized insights based on your {persona || 'profile'}...
        </p>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-primary-400" />
        <h2 className="text-lg font-semibold text-white">AI Insights for You</h2>
        {persona && (
          <span className="px-2 py-1 text-xs rounded-full bg-primary-500/20 text-primary-400 capitalize">
            {persona}
          </span>
        )}
      </div>
      
      <div className="space-y-4">
        {insights.map((insight, index) => (
          <AIInsightCard key={insight.id || index} insight={insight} index={index} />
        ))}
      </div>
    </motion.div>
  )
}

export default AIInsightsList
