/**
 * Skeleton shimmer loaders for various card shapes.
 */
export function SkeletonCard({ className = "" }) {
  return (
    <div className={`skeleton h-48 rounded-3xl ${className}`} />
  );
}

export function SkeletonHourly() {
  return (
    <div className="skeleton h-28 rounded-3xl" />
  );
}

export function SkeletonForecast() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="skeleton h-12 rounded-xl" />
      ))}
    </div>
  );
}

export default function Loader({ text = "Loading weather data..." }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 rounded-full border-2 border-indigo-500/20" />
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-indigo-400 animate-spin" />
        <span className="absolute inset-0 flex items-center justify-center text-2xl">⛅</span>
      </div>
      <p className="text-gray-500 text-sm">{text}</p>
    </div>
  );
}
