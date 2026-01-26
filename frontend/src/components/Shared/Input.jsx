import { forwardRef } from 'react'
import clsx from 'clsx'

const Input = forwardRef(({ 
  label,
  error,
  icon: Icon,
  rightIcon: RightIcon,
  onRightIconClick,
  className = '',
  containerClassName = '',
  ...props 
}, ref) => {
  return (
    <div className={clsx('space-y-2', containerClassName)}>
      {label && (
        <label className="block text-sm font-medium text-white/70">
          {label}
        </label>
      )}
      <div className="relative">
        {Icon && (
          <div className="absolute left-4 top-1/2 -translate-y-1/2 text-white/40">
            <Icon className="w-5 h-5" />
          </div>
        )}
        <input
          ref={ref}
          className={clsx(
            'w-full bg-white/5 backdrop-blur-sm border rounded-xl px-4 py-3 text-white placeholder-white/40',
            'focus:outline-none focus:border-primary-500/50 focus:ring-2 focus:ring-primary-500/20',
            'transition-all duration-300',
            Icon && 'pl-12',
            RightIcon && 'pr-12',
            error ? 'border-weather-storm/50' : 'border-white/10',
            className
          )}
          {...props}
        />
        {RightIcon && (
          <button
            type="button"
            onClick={onRightIconClick}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/70 transition-colors"
          >
            <RightIcon className="w-5 h-5" />
          </button>
        )}
      </div>
      {error && (
        <p className="text-sm text-weather-storm">{error}</p>
      )}
    </div>
  )
})

Input.displayName = 'Input'

export default Input
