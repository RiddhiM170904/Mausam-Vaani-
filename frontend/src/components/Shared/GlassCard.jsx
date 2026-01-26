import { motion } from 'framer-motion'
import clsx from 'clsx'

const GlassCard = ({ 
  children, 
  className = '', 
  hover = false, 
  neonBorder = false,
  accentColor = null, // 'sunny' | 'rain' | 'storm' | 'primary'
  onClick = null,
  animate = true 
}) => {
  const baseClasses = 'bg-white/10 backdrop-blur-lg border rounded-2xl'
  
  const borderColor = accentColor 
    ? {
        sunny: 'border-weather-sunny/50',
        rain: 'border-weather-rain/50',
        storm: 'border-weather-storm/50',
        primary: 'border-primary-500/50',
      }[accentColor]
    : 'border-white/20'

  const glowEffect = accentColor
    ? {
        sunny: 'shadow-[0_0_30px_rgba(251,191,36,0.2)]',
        rain: 'shadow-[0_0_30px_rgba(34,211,238,0.2)]',
        storm: 'shadow-[0_0_30px_rgba(239,68,68,0.2)]',
        primary: 'shadow-neon',
      }[accentColor]
    : ''

  const hoverClasses = hover 
    ? 'transition-all duration-300 hover:bg-white/15 hover:border-white/30 hover:shadow-neon cursor-pointer' 
    : ''

  const Component = animate ? motion.div : 'div'
  const animationProps = animate ? {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.4 },
    whileHover: hover ? { scale: 1.02 } : {}
  } : {}

  return (
    <Component
      className={clsx(baseClasses, borderColor, glowEffect, hoverClasses, className)}
      onClick={onClick}
      {...animationProps}
    >
      {children}
    </Component>
  )
}

export default GlassCard
