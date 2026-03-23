import { motion } from "framer-motion";
import GlassCard from "./GlassCard";
import { formatTemperatureC, formatWindKmh, getOWMIconUrl } from "../utils/helpers";

/**
 * Main current-weather hero card.
 */
export default function WeatherCard({ data, city }) {
  if (!data) return null;

  const { temp, feelsLike, condition, description, icon, humidity, wind, aqiUs } = data;

  const getAqiCondition = (value) => {
    const aqi = Number(value);
    if (!Number.isFinite(aqi)) return "Unknown";
    if (aqi <= 50) return "Good";
    if (aqi <= 100) return "Moderate";
    if (aqi <= 150) return "Unhealthy for Sensitive";
    if (aqi <= 200) return "Unhealthy";
    if (aqi <= 300) return "Very Unhealthy";
    return "Critical";
  };

  const aqiText = `${aqiUs ?? "--"} ${getAqiCondition(aqiUs)}`;

  return (
    <GlassCard className="relative p-6 overflow-hidden sm:p-8">
      {/* Background glow */}
      <div className="absolute rounded-full pointer-events-none -top-20 -right-20 w-60 h-60 bg-indigo-500/20 blur-3xl" />
      <div className="absolute w-48 h-48 rounded-full pointer-events-none -bottom-20 -left-20 bg-purple-500/15 blur-3xl" />

      <div className="relative z-10">
        {/* City */}
        <p className="mb-1 text-sm font-medium tracking-wide text-gray-400 uppercase">
          {city || "Loading..."}
        </p>
        <p className="mb-4 text-xs text-gray-500 capitalize">{description}</p>

        <div className="flex items-center justify-between">
          {/* Temperature */}
          <div>
            <motion.h1
              className="tracking-tighter text-white text-7xl sm:text-8xl font-extralight"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            >
              {formatTemperatureC(temp)}
            </motion.h1>
            <p className="mt-2 text-sm text-gray-400">
              Feels like {formatTemperatureC(feelsLike)} · {condition}
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
          <span>💨 {formatWindKmh(wind)}</span>
          <span className="inline-flex items-center gap-1 text-emerald-300">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.75)]"></span>
            AQI - {aqiText}
          </span>
        </div>
      </div>
    </GlassCard>
  );
}
