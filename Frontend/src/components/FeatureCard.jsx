import clsx from 'clsx'

const FeatureCard = ({ icon: Icon, title, description, innovation, className }) => {
  return (
    <div className={clsx(
      "bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300 border border-gray-100",
      className
    )}>
      <div className="flex items-center justify-center w-14 h-14 bg-primary-100 rounded-lg mb-4">
        <Icon className="h-8 w-8 text-primary-600" />
      </div>
      <h3 className="text-xl font-bold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600 mb-3">{description}</p>
      {innovation && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <span className="text-xs font-semibold text-primary-600 uppercase tracking-wide">Innovation</span>
          <p className="text-sm text-gray-700 mt-1">{innovation}</p>
        </div>
      )}
    </div>
  )
}

export default FeatureCard
