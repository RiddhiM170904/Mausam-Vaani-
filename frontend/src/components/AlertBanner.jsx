import { motion, AnimatePresence } from "framer-motion";
import { HiExclamationTriangle, HiXMark } from "react-icons/hi2";
import { useState } from "react";

/**
 * Top alert banner for weather warnings.
 */
export default function AlertBanner({ alerts = [] }) {
  const [dismissed, setDismissed] = useState(new Set());

  const visible = alerts.filter((_, i) => !dismissed.has(i));
  if (!visible.length) return null;

  const dismiss = (idx) => {
    setDismissed((prev) => new Set([...prev, idx]));
  };

  const severityColors = {
    extreme: "from-red-600/30 to-red-900/20 border-red-500/40",
    severe: "from-red-600/20 to-red-900/10 border-red-500/30",
    warning: "from-yellow-600/20 to-yellow-900/10 border-yellow-500/30",
    watch: "from-orange-600/20 to-orange-900/10 border-orange-500/30",
    advisory: "from-blue-600/20 to-blue-900/10 border-blue-500/30",
  };

  return (
    <AnimatePresence>
      {alerts.map((alert, i) =>
        dismissed.has(i) ? null : (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: -20, height: 0 }}
            animate={{ opacity: 1, y: 0, height: "auto" }}
            exit={{ opacity: 0, y: -20, height: 0 }}
            className={`rounded-2xl bg-gradient-to-r ${
              severityColors[alert.severity] || severityColors.warning
            } border backdrop-blur-xl p-4 flex items-start gap-3`}
          >
            <HiExclamationTriangle className="text-yellow-400 flex-shrink-0 mt-0.5" size={20} />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-white">{alert.event}</p>
              <p className="text-xs text-gray-300 mt-1 line-clamp-2">{alert.description}</p>
            </div>
            <button
              onClick={() => dismiss(i)}
              className="p-1 rounded-lg hover:bg-white/[0.1] transition-colors text-gray-400 hover:text-white flex-shrink-0"
            >
              <HiXMark size={16} />
            </button>
          </motion.div>
        )
      )}
    </AnimatePresence>
  );
}
