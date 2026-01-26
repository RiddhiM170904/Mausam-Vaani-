import clsx from 'clsx'

const FeatureCard = ({ icon: Icon, title, description, innovation, className }) => {
  return (
    <div className={clsx(
      "glass-card p-6 hover:scale-[1.02] transition-all duration-300",
      className
    )}>
      <div className="flex items-center justify-center w-14 h-14 bg-primary-500/20 rounded-xl mb-4 border border-primary-500/30">
        <Icon className="h-8 w-8 text-primary-400" />
      </div>
      <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
      <p className="text-white/70 mb-3">{description}</p>
      {innovation && (
        <div className="mt-4 pt-4 border-t border-white/10">
          <span className="text-xs font-semibold text-primary-400 uppercase tracking-wide">Innovation</span>
          <p className="text-sm text-white/60 mt-1">{innovation}</p>
        </div>
      )}
    </div>
  )
}

export default FeatureCard
