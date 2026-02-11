import { motion } from "framer-motion";
import GlassCard from "./GlassCard";
import { getOWMIconUrl } from "../utils/helpers";

/**
 * Main current-weather hero card.
 */
export default function WeatherCard({ data, city }) {
  if (!data) return null;

  const { temp, feelsLike, condition, description, icon, humidity, wind } = data;

  return (
    <GlassCard className="p-6 sm:p-8 relative overflow-hidden">
      {/* Background glow */}
      <div className="absolute -top-20 -right-20 w-60 h-60 bg-indigo-500/20 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-20 -left-20 w-48 h-48 bg-purple-500/15 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10">
        {/* City */}
        <p className="text-gray-400 text-sm font-medium tracking-wide uppercase mb-1">
          {city || "Loading..."}
        </p>
        <p className="text-gray-500 text-xs mb-4 capitalize">{description}</p>

        <div className="flex items-center justify-between">
          {/* Temperature */}
          <div>
            <motion.h1
              className="text-7xl sm:text-8xl font-extralight text-white tracking-tighter"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            >
              {temp}°
            </motion.h1>
            <p className="text-gray-400 text-sm mt-2">
              Feels like {feelsLike}° · {condition}
            </p>
          </div>

          {/* Weather Icon */}
          <motion.img
            src={getOWMIconUrl(icon, 4)}
            alt={condition}
            className="w-28 h-28 sm:w-36 sm:h-36 drop-shadow-2xl"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          />
        </div>

        {/* Quick stats */}
        <div className="flex gap-6 mt-6 text-sm text-gray-400">
          <span>💧 {humidity}%</span>
          <span>💨 {wind} km/h</span>
        </div>
      </div>
    </GlassCard>
  );
}
