import { motion } from "framer-motion";
import GlassCard from "./GlassCard";
import { formatDay, getOWMIconUrl } from "../utils/helpers";

/**
 * 7-day forecast list.
 */
export default function ForecastCard({ days = [] }) {
  if (!days.length) return null;

  return (
    <GlassCard className="p-4 sm:p-6" hover={false}>
      <h3 className="text-sm font-semibold text-gray-300 mb-4 px-1">7-Day Forecast</h3>
      <div className="space-y-1">
        {days.map((day, i) => (
          <motion.div
            key={day.date}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.06 }}
            className="flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/[0.04] transition-colors"
          >
            <span className="text-sm text-gray-300 w-24 font-medium">
              {formatDay(day.date)}
            </span>
            <div className="flex items-center gap-2 flex-1 justify-center">
              <img
                src={getOWMIconUrl(day.icon)}
                alt={day.condition}
                className="w-8 h-8"
              />
              <span className="text-xs text-gray-500 hidden sm:block">{day.condition}</span>
            </div>
            <div className="text-sm text-right">
              <span className="text-white font-semibold">{day.tempMax}°</span>
              <span className="text-gray-500 ml-2">{day.tempMin}°</span>
            </div>
          </motion.div>
        ))}
      </div>
    </GlassCard>
  );
}
