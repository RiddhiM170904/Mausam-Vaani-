import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useAuth } from "../context/AuthContext";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import { insightService } from "../services/insightService";
import GlassCard from "../components/GlassCard";
import Loader from "../components/Loader";
import {
  HiExclamationTriangle,
  HiShieldCheck,
  HiInformationCircle,
} from "react-icons/hi2";

const AI_ALERT_STORAGE_KEY = "mv_ai_alert_updates_v1";
const AI_ALERT_INTERVAL_MS = 2 * 60 * 60 * 1000;
const AI_ALERT_TTL_MS = 12 * 60 * 60 * 1000;
const MAX_AI_ALERTS = 3;

function toAlertMessage(payload) {
  if (!payload) return "";
  if (typeof payload === "string") return payload;
  if (typeof payload?.message === "string" && payload.message.trim()) return payload.message.trim();
  if (typeof payload?.recommendation === "string" && payload.recommendation.trim()) return payload.recommendation.trim();
  if (Array.isArray(payload?.tips) && payload.tips.length > 0) return String(payload.tips[0]);
  return "";
}

function pruneAiAlerts(list, userId = null) {
  const now = Date.now();
  const safeList = Array.isArray(list) ? list : [];
  return safeList.filter((item) => {
    const createdAt = new Date(item?.generatedAt || 0).getTime();
    if (!Number.isFinite(createdAt)) return false;
    if (now - createdAt > AI_ALERT_TTL_MS) return false;
    if (userId && item?.userId && String(item.userId) !== String(userId)) return false;
    return true;
  });
}

function readAiAlerts(userId = null) {
  try {
    const raw = localStorage.getItem(AI_ALERT_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    const pruned = pruneAiAlerts(parsed, userId);
    localStorage.setItem(AI_ALERT_STORAGE_KEY, JSON.stringify(pruned));
    return pruned;
  } catch {
    return [];
  }
}

function writeAiAlerts(list) {
  try {
    localStorage.setItem(AI_ALERT_STORAGE_KEY, JSON.stringify(list));
  } catch {
    // Ignore localStorage failures.
  }
}

/**
 * Weather alerts page — shows all active warnings.
 */
export default function Alerts() {
  const { isLoggedIn, user } = useAuth();
  const { location } = useLocation();
  const { data, loading } = useWeather(location?.lat, location?.lon);
  const [aiUpdates, setAiUpdates] = useState([]);
  const [aiUpdating, setAiUpdating] = useState(false);

  const weatherData = data;

  useEffect(() => {
    if (!isLoggedIn) {
      setAiUpdates([]);
      return;
    }

    setAiUpdates(readAiAlerts(user?.id || null).slice(0, MAX_AI_ALERTS));
  }, [isLoggedIn, user?.id]);

  useEffect(() => {
    if (!isLoggedIn || !weatherData) return;

    let cancelled = false;

    const runAiUpdate = async () => {
      setAiUpdating(true);
      try {
        const result = await insightService.getQuickInsight({
          userId: user?.id || null,
          persona: user?.persona || "general",
          weatherRisks: user?.weather_risks || user?.weatherRisks || [],
          weatherData,
          location,
          requirements:
            "Generate short alert-style Hinglish guidance for next 2 hours with clear do/avoid action.",
        });

        if (cancelled) return;

        const message = toAlertMessage(result);
        if (!message) return;

        setAiUpdates((prev) => {
          const current = pruneAiAlerts(prev, user?.id || null);
          const latest = current[0];

          // Avoid stacking duplicates on every cycle.
          if (latest && String(latest.message || "").trim() === message.trim()) {
            return current;
          }

          const next = [
            {
              id: `${Date.now()}`,
              message,
              generatedAt: new Date().toISOString(),
              source: result?.source || "ai",
              userId: user?.id || null,
            },
            ...current,
          ].slice(0, MAX_AI_ALERTS);

          writeAiAlerts(next);
          return next;
        });
      } catch {
        // Keep real-time weather alerts unaffected if AI update fails.
      } finally {
        if (!cancelled) setAiUpdating(false);
      }
    };

    runAiUpdate();

    const intervalId = setInterval(() => {
      if (document.visibilityState === "visible") {
        runAiUpdate();
      }
    }, AI_ALERT_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [isLoggedIn, user?.id, user?.persona, user?.weather_risks, user?.weatherRisks, weatherData, location?.lat, location?.lon, location?.city]);

  const aiAlerts = useMemo(() => aiUpdates.slice(0, MAX_AI_ALERTS), [aiUpdates]);
  const createdAlerts = useMemo(
    () =>
      aiAlerts.map((item) => ({
        event: "Notification",
        description: item.message,
        severity: "advisory",
        generatedAt: item.generatedAt,
      })),
    [aiAlerts]
  );

  if (loading) return <Loader text="Checking alerts..." />;

  const weatherAlerts = data?.alerts || [];
  const alerts = [...weatherAlerts, ...createdAlerts];

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
        <div className="space-y-3 px-1 sm:px-0">
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
                      {alert.generatedAt && (
                        <p className="text-xs text-gray-600 mt-2">
                          Generated: {new Date(alert.generatedAt).toLocaleString()}
                        </p>
                      )}
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
