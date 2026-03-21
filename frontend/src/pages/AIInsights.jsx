import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, TriangleAlert, CalendarCheck2, ShieldCheck, Wind, Thermometer, CloudRain } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import { insightService } from "../services/insightService";
import GlassCard from "../components/GlassCard";
import Loader from "../components/Loader";

function buildActionItems(weather) {
  const current = weather?.current || {};
  const temp = Number(current.temp || 0);
  const humidity = Number(current.humidity || 0);
  const wind = Number(current.wind || 0);
  const condition = String(current.condition || "").toLowerCase();

  const actions = [];

  if (temp >= 35) actions.push("Avoid heavy outdoor activity between 12 PM and 4 PM.");
  if (condition.includes("rain")) actions.push("Keep rain protection ready and start commute earlier.");
  if (wind >= 20) actions.push("Expect stronger winds. Secure loose items and drive cautiously.");
  if (humidity >= 80) actions.push("Plan hydration breaks if spending long time outdoors.");
  if (!actions.length) actions.push("Weather looks stable. Continue with normal plans and monitor updates.");

  return actions;
}

export default function AIInsights() {
  const { user, isLoggedIn } = useAuth();
  const { location } = useLocation();
  const { data: weatherData, loading: weatherLoading } = useWeather(location?.lat, location?.lon);
  const [insightLoading, setInsightLoading] = useState(false);
  const [insight, setInsight] = useState(null);

  useEffect(() => {
    if (!weatherData) return;

    setInsightLoading(true);
    insightService
      .getQuickInsight({
        weatherData,
        persona: user?.persona || "general",
        weatherRisks: user?.weather_risks || [],
        location,
      })
      .then((res) => setInsight(res))
      .finally(() => setInsightLoading(false));
  }, [weatherData, user?.persona, user?.weather_risks, location]);

  const actionItems = useMemo(() => buildActionItems(weatherData), [weatherData]);

  if (weatherLoading) return <Loader text="Generating AI insights..." />;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mx-auto w-full max-w-4xl space-y-4 sm:space-y-6">
      <div className="space-y-1 px-1">
        <h1 className="text-2xl font-bold text-white">AI Insights</h1>
        <p className="text-sm text-gray-400">
          Decision-focused weather intelligence for {isLoggedIn ? user?.name || "you" : "your day"}.
        </p>
      </div>

      <GlassCard className="border-indigo-500/20 bg-linear-to-r from-indigo-500/10 to-blue-500/10 p-5 sm:p-6">
        <div className="flex items-start gap-3 sm:gap-4">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-500/20">
            <Sparkles className="h-5 w-5 text-indigo-300" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-indigo-300">Personalized Recommendation</p>
            {insightLoading ? (
              <p className="mt-2 text-sm text-gray-400">Analyzing your weather and risk profile...</p>
            ) : (
              <p className="mt-2 text-sm leading-relaxed text-gray-200">{insight?.message || "No insight available yet."}</p>
            )}
          </div>
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <GlassCard className="p-4" hover={false}>
          <div className="mb-2 flex items-center gap-2 text-red-300">
            <TriangleAlert className="h-4 w-4" />
            <p className="text-xs font-semibold uppercase tracking-wide">Risk Snapshot</p>
          </div>
          <p className="text-sm text-gray-300">
            Temp: {weatherData?.current?.temp ?? "--"}°C, Wind: {weatherData?.current?.wind ?? "--"} km/h
          </p>
          <p className="mt-1 text-sm text-gray-500 capitalize">Condition: {weatherData?.current?.condition || "Unknown"}</p>
        </GlassCard>

        <GlassCard className="p-4" hover={false}>
          <div className="mb-2 flex items-center gap-2 text-emerald-300">
            <CalendarCheck2 className="h-4 w-4" />
            <p className="text-xs font-semibold uppercase tracking-wide">Best Window</p>
          </div>
          <p className="text-sm text-gray-300">Morning and evening windows are generally safer for outdoor tasks.</p>
          <p className="mt-1 text-sm text-gray-500">Use planner for activity-level timing.</p>
        </GlassCard>

        <GlassCard className="p-4" hover={false}>
          <div className="mb-2 flex items-center gap-2 text-cyan-300">
            <ShieldCheck className="h-4 w-4" />
            <p className="text-xs font-semibold uppercase tracking-wide">Preparedness</p>
          </div>
          <p className="text-sm text-gray-300">Keep essentials based on today: hydration, mask, rain cover, or light layers.</p>
        </GlassCard>
      </div>

      <GlassCard className="p-5 sm:p-6" hover={false}>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-gray-300">Action Plan For Today</h2>
        <div className="space-y-3">
          {actionItems.map((item, idx) => (
            <div key={idx} className="flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 p-3">
              <span className="mt-0.5 text-indigo-300">•</span>
              <p className="text-sm text-gray-200">{item}</p>
            </div>
          ))}
        </div>
      </GlassCard>

      <div className="grid grid-cols-3 gap-3 sm:gap-4">
        <GlassCard className="p-3 text-center" hover={false}>
          <CloudRain className="mx-auto h-4 w-4 text-blue-300" />
          <p className="mt-2 text-xs text-gray-400">Humidity</p>
          <p className="text-sm font-semibold text-white">{weatherData?.current?.humidity ?? "--"}%</p>
        </GlassCard>
        <GlassCard className="p-3 text-center" hover={false}>
          <Thermometer className="mx-auto h-4 w-4 text-orange-300" />
          <p className="mt-2 text-xs text-gray-400">Feels Like</p>
          <p className="text-sm font-semibold text-white">{weatherData?.current?.feelsLike ?? "--"}°</p>
        </GlassCard>
        <GlassCard className="p-3 text-center" hover={false}>
          <Wind className="mx-auto h-4 w-4 text-cyan-300" />
          <p className="mt-2 text-xs text-gray-400">Wind</p>
          <p className="text-sm font-semibold text-white">{weatherData?.current?.wind ?? "--"} km/h</p>
        </GlassCard>
      </div>
    </motion.div>
  );
}
