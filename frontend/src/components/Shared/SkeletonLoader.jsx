import clsx from 'clsx'

const SkeletonLoader = ({ 
  variant = 'text', // 'text' | 'circle' | 'rect' | 'card'
  width,
  height,
  className = '',
  count = 1,
}) => {
  const variants = {
    text: 'h-4 rounded',
    circle: 'rounded-full',
    rect: 'rounded-xl',
    card: 'h-32 rounded-2xl',
  }

  const items = Array.from({ length: count }, (_, i) => (
    <div
      key={i}
      className={clsx(
        'skeleton',
        variants[variant],
        className
      )}
      style={{
        width: width || (variant === 'circle' ? '40px' : '100%'),
        height: height || (variant === 'circle' ? '40px' : undefined),
      }}
    />
  ))

  return count === 1 ? items[0] : <div className="space-y-3">{items}</div>
}

// Weather Card Skeleton
export const WeatherCardSkeleton = () => (
  <div className="glass-card p-6 space-y-4">
    <div className="flex items-center gap-4">
      <SkeletonLoader variant="circle" width="64px" height="64px" />
      <div className="flex-1 space-y-2">
        <SkeletonLoader width="60%" />
        <SkeletonLoader width="40%" />
      </div>
    </div>
    <SkeletonLoader variant="rect" height="100px" />
    <div className="flex gap-2">
      <SkeletonLoader width="25%" />
      <SkeletonLoader width="25%" />
      <SkeletonLoader width="25%" />
      <SkeletonLoader width="25%" />
    </div>
  </div>
)

// Hourly Forecast Skeleton
export const HourlyForecastSkeleton = () => (
  <div className="flex gap-4 overflow-hidden">
    {Array.from({ length: 6 }).map((_, i) => (
      <div key={i} className="glass-card p-4 min-w-[80px] space-y-2">
        <SkeletonLoader width="40px" className="mx-auto" />
        <SkeletonLoader variant="circle" width="32px" height="32px" className="mx-auto" />
        <SkeletonLoader width="30px" className="mx-auto" />
      </div>
    ))}
  </div>
)

// Insight Card Skeleton
export const InsightCardSkeleton = () => (
  <div className="glass-card p-6 space-y-4 border-l-4 border-primary-500/50">
    <div className="flex items-center gap-3">
      <SkeletonLoader variant="circle" width="40px" height="40px" />
      <SkeletonLoader width="150px" />
    </div>
    <SkeletonLoader count={3} />
  </div>
)

export default SkeletonLoader
