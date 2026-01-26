import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  MessageSquare, ThumbsUp, Share2, MapPin, Clock, 
  AlertTriangle, Cloud, CheckCircle, Send, Image, Plus
} from 'lucide-react'
import { GlassCard, Button, Input } from '../components/Shared'

const Community = () => {
  const [newPost, setNewPost] = useState('')

  const posts = [
    {
      id: 1,
      user: 'Ramesh Kumar',
      location: 'Sector 4, Bhopal',
      time: '15 min ago',
      type: 'alert',
      content: 'Heavy hail happening right now! Stay indoors. 🌨️',
      verified: true,
      likes: 24,
      comments: 8,
    },
    {
      id: 2,
      user: 'Sita Devi',
      location: 'Indore Rural',
      time: '1 hour ago',
      type: 'update',
      content: 'Light drizzle started. Good for crops after the dry spell.',
      verified: false,
      likes: 12,
      comments: 3,
    },
    {
      id: 3,
      user: 'IMD Bhopal',
      location: 'Official',
      time: '2 hours ago',
      type: 'official',
      content: 'Weather Advisory: Western disturbance expected to bring rain to MP over next 48 hours. Farmers advised to complete harvesting operations.',
      verified: true,
      likes: 156,
      comments: 42,
    },
    {
      id: 4,
      user: 'Kishan Singh',
      location: 'Kothri Kalan',
      time: '3 hours ago',
      type: 'report',
      content: 'Fog cleared by 9 AM. Visibility is good now. Safe for travel.',
      verified: false,
      likes: 8,
      comments: 2,
    },
  ]

  const getTypeStyles = (type) => {
    switch (type) {
      case 'alert':
        return { bg: 'bg-weather-storm/10', border: 'border-weather-storm/30', icon: AlertTriangle, color: 'text-weather-storm' }
      case 'official':
        return { bg: 'bg-primary-500/10', border: 'border-primary-500/30', icon: CheckCircle, color: 'text-primary-400' }
      case 'update':
        return { bg: 'bg-weather-rain/10', border: 'border-weather-rain/30', icon: Cloud, color: 'text-weather-rain' }
      default:
        return { bg: 'bg-white/5', border: 'border-white/10', icon: MessageSquare, color: 'text-white/60' }
    }
  }

  return (
    <div className="min-h-screen py-6">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-display font-bold text-white mb-2">Community</h1>
          <p className="text-white/60">Live weather reports from your area</p>
        </motion.div>

        {/* Post Input */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-6"
        >
          <GlassCard className="p-4">
            <div className="flex gap-4">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-primary-400 font-medium">U</span>
              </div>
              <div className="flex-1">
                <textarea
                  value={newPost}
                  onChange={(e) => setNewPost(e.target.value)}
                  placeholder="Share a weather update from your location..."
                  className="w-full bg-transparent text-white placeholder-white/40 resize-none focus:outline-none"
                  rows={2}
                />
                <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/10">
                  <div className="flex items-center gap-2">
                    <button className="p-2 rounded-lg hover:bg-white/10 transition-colors">
                      <MapPin className="w-5 h-5 text-primary-400" />
                    </button>
                    <button className="p-2 rounded-lg hover:bg-white/10 transition-colors">
                      <Image className="w-5 h-5 text-white/60" />
                    </button>
                    <span className="text-xs text-white/40 ml-2">
                      <MapPin className="w-3 h-3 inline mr-1" />
                      Kothri Kalan, MP
                    </span>
                  </div>
                  <Button 
                    variant="primary" 
                    size="sm" 
                    icon={Send}
                    disabled={!newPost.trim()}
                  >
                    Post
                  </Button>
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>

        {/* Feed */}
        <div className="space-y-4">
          {posts.map((post, index) => {
            const styles = getTypeStyles(post.type)
            const Icon = styles.icon

            return (
              <motion.div
                key={post.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + index * 0.05 }}
              >
                <GlassCard className={`p-4 ${styles.bg} ${styles.border} border`}>
                  {/* Header */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center">
                        <span className="text-white font-medium">{post.user[0]}</span>
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-white">{post.user}</span>
                          {post.verified && (
                            <CheckCircle className="w-4 h-4 text-primary-400" />
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-white/50">
                          <MapPin className="w-3 h-3" />
                          <span>{post.location}</span>
                          <span>•</span>
                          <Clock className="w-3 h-3" />
                          <span>{post.time}</span>
                        </div>
                      </div>
                    </div>
                    <div className={`p-2 rounded-lg ${styles.bg}`}>
                      <Icon className={`w-4 h-4 ${styles.color}`} />
                    </div>
                  </div>

                  {/* Content */}
                  <p className="text-white/80 leading-relaxed mb-4">{post.content}</p>

                  {/* Actions */}
                  <div className="flex items-center gap-4 pt-3 border-t border-white/10">
                    <button className="flex items-center gap-2 text-white/50 hover:text-primary-400 transition-colors">
                      <ThumbsUp className="w-4 h-4" />
                      <span className="text-sm">{post.likes}</span>
                    </button>
                    <button className="flex items-center gap-2 text-white/50 hover:text-primary-400 transition-colors">
                      <MessageSquare className="w-4 h-4" />
                      <span className="text-sm">{post.comments}</span>
                    </button>
                    <button className="flex items-center gap-2 text-white/50 hover:text-green-400 transition-colors ml-auto">
                      <Share2 className="w-4 h-4" />
                      <span className="text-sm hidden sm:inline">Share</span>
                    </button>
                  </div>
                </GlassCard>
              </motion.div>
            )
          })}
        </div>

        {/* Load More */}
        <div className="text-center mt-8">
          <Button variant="secondary" icon={Plus}>
            Load More Updates
          </Button>
        </div>
      </div>
    </div>
  )
}

export default Community
