import { useRef } from "react";
import { motion } from "framer-motion";
import GlassCard from "./GlassCard";
import { getOWMIconUrl } from "../utils/helpers";

/**
 * Horizontally scrollable hourly forecast strip.
 */
export default function HourlySlider({ hours = [] }) {
  const scrollRef = useRef(null);

  if (!hours.length) return null;

  return (
    <GlassCard className="p-4" hover={false}>
      <h3 className="text-sm font-semibold text-gray-300 mb-3 px-1">Hourly Forecast</h3>
      <div
        ref={scrollRef}
        className="flex gap-3 overflow-x-auto hide-scrollbar pb-1"
      >
        {hours.map((h, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="flex-shrink-0 flex flex-col items-center gap-1 px-4 py-3 rounded-2xl bg-white/[0.04] hover:bg-white/[0.08] transition-colors min-w-[72px]"
          >
            <span className="text-xs text-gray-400 font-medium">{h.time}</span>
            <img
              src={getOWMIconUrl(h.icon)}
              alt={h.condition}
              className="w-8 h-8"
            />
            <span className="text-sm font-semibold text-white">{h.temp}°</span>
          </motion.div>
        ))}
      </div>
    </GlassCard>
  );
}
