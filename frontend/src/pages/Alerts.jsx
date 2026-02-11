import { motion } from "framer-motion";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import GlassCard from "../components/GlassCard";
import Loader from "../components/Loader";
import {
  HiExclamationTriangle,
  HiShieldCheck,
  HiInformationCircle,
} from "react-icons/hi2";

/**
 * Weather alerts page — shows all active warnings.
 */
export default function Alerts() {
  const { location } = useLocation();
  const { data, loading } = useWeather(location?.lat, location?.lon);

  if (loading) return <Loader text="Checking alerts..." />;

  const alerts = data?.alerts || [];

  const severityConfig = {
    extreme: {
      icon: <HiExclamationTriangle size={24} />,
      color: "text-red-400",
      bg: "bg-red-500/10 border-red-500/20",
    },
    severe: {
      icon: <HiExclamationTriangle size={24} />,
      color: "text-red-400",
      bg: "bg-red-500/10 border-red-500/20",
    },
    warning: {
      icon: <HiExclamationTriangle size={24} />,
      color: "text-yellow-400",
      bg: "bg-yellow-500/10 border-yellow-500/20",
    },
    watch: {
      icon: <HiInformationCircle size={24} />,
      color: "text-orange-400",
      bg: "bg-orange-500/10 border-orange-500/20",
    },
    advisory: {
      icon: <HiInformationCircle size={24} />,
      color: "text-blue-400",
      bg: "bg-blue-500/10 border-blue-500/20",
    },
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      <h1 className="text-2xl font-bold text-white">Weather Alerts</h1>

      {alerts.length === 0 ? (
        <GlassCard className="p-8 text-center">
          <HiShieldCheck size={48} className="text-green-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">All Clear</h3>
          <p className="text-gray-500 text-sm">
            No active weather alerts for your area. Stay safe!
          </p>
        </GlassCard>
      ) : (
        <div className="space-y-3">
          {alerts.map((alert, i) => {
            const config =
              severityConfig[alert.severity] || severityConfig.warning;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
              >
                <GlassCard
                  className={`p-5 border ${config.bg}`}
                  hover={false}
                >
                  <div className="flex items-start gap-4">
                    <div className={`mt-1 ${config.color}`}>{config.icon}</div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-white">{alert.event}</h3>
                        <span
                          className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full ${config.bg} ${config.color}`}
                        >
                          {alert.severity || "warning"}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400 leading-relaxed">
                        {alert.description}
                      </p>
                      {alert.start && (
                        <p className="text-xs text-gray-600 mt-2">
                          {new Date(alert.start * 1000).toLocaleString()} —{" "}
                          {new Date(alert.end * 1000).toLocaleString()}
                        </p>
                      )}
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Safety tips */}
      <GlassCard className="p-5" hover={false}>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Safety Tips</h3>
        <ul className="space-y-2 text-sm text-gray-500">
          <li className="flex items-start gap-2">
            <span>🌊</span>
            <span>During floods, move to higher ground immediately</span>
          </li>
          <li className="flex items-start gap-2">
            <span>🌡️</span>
            <span>In heat waves, stay hydrated and avoid outdoor activities 12-3pm</span>
          </li>
          <li className="flex items-start gap-2">
            <span>⚡</span>
            <span>During thunderstorms, stay indoors and unplug electronics</span>
          </li>
          <li className="flex items-start gap-2">
            <span>🌪️</span>
            <span>In cyclone warnings, follow government evacuation orders</span>
          </li>
        </ul>
      </GlassCard>
    </motion.div>
  );
}
