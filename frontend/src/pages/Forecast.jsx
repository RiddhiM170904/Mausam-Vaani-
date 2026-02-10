import { useState } from "react";
import { motion } from "framer-motion";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import Loader from "../components/Loader";
import GlassCard from "../components/GlassCard";
import { getOWMIconUrl, formatDay } from "../utils/helpers";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function Forecast() {
  const { location } = useLocation();
  const { data, loading } = useWeather(location?.lat, location?.lon);
  const [view, setView] = useState("hourly"); // hourly | daily

  if (loading) return <Loader text="Loading forecast..." />;
  if (!data) return null;

  const hourlyChartData = (data.hourly || []).map((h) => ({
    time: h.time,
    temp: h.temp,
  }));

  const dailyChartData = (data.daily || []).map((d) => ({
    day: formatDay(d.date),
    max: d.tempMax,
    min: d.tempMin,
  }));

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Forecast</h1>
        <div className="flex gap-1 p-1 rounded-xl bg-white/[0.06]">
          {["hourly", "daily"].map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                view === v
                  ? "bg-indigo-500/30 text-indigo-300"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <GlassCard className="p-4 sm:p-6" hover={false}>
        <h3 className="text-sm font-semibold text-gray-400 mb-4">
          {view === "hourly" ? "Temperature Trend (Next 24h)" : "Weekly Temperature Range"}
        </h3>
        <div className="h-52 sm:h-64">
          <ResponsiveContainer width="100%" height="100%">
            {view === "hourly" ? (
              <AreaChart data={hourlyChartData}>
                <defs>
                  <linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#818cf8" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="time"
                  stroke="#4b5563"
                  tick={{ fontSize: 11, fill: "#6b7280" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  stroke="#4b5563"
                  tick={{ fontSize: 11, fill: "#6b7280" }}
                  axisLine={false}
                  tickLine={false}
                  domain={["dataMin - 2", "dataMax + 2"]}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(15,23,42,0.9)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "12px",
                    color: "#fff",
                    fontSize: "13px",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="temp"
                  stroke="#818cf8"
                  strokeWidth={2.5}
                  fill="url(#tempGrad)"
                />
              </AreaChart>
            ) : (
              <AreaChart data={dailyChartData}>
                <defs>
                  <linearGradient id="maxGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f97316" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#f97316" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="minGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="day"
                  stroke="#4b5563"
                  tick={{ fontSize: 11, fill: "#6b7280" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  stroke="#4b5563"
                  tick={{ fontSize: 11, fill: "#6b7280" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(15,23,42,0.9)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "12px",
                    color: "#fff",
                    fontSize: "13px",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="max"
                  stroke="#f97316"
                  strokeWidth={2}
                  fill="url(#maxGrad)"
                />
                <Area
                  type="monotone"
                  dataKey="min"
                  stroke="#38bdf8"
                  strokeWidth={2}
                  fill="url(#minGrad)"
                />
              </AreaChart>
            )}
          </ResponsiveContainer>
        </div>
      </GlassCard>

      {/* Detailed list */}
      {view === "hourly" ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {(data.hourly || []).map((h, i) => (
            <GlassCard key={i} className="p-4 flex flex-col items-center gap-2">
              <span className="text-xs text-gray-400">{h.time}</span>
              <img src={getOWMIconUrl(h.icon)} alt="" className="w-10 h-10" />
              <span className="text-lg font-bold text-white">{h.temp}°</span>
              <span className="text-xs text-gray-500">{h.condition}</span>
            </GlassCard>
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {(data.daily || []).map((day, i) => (
            <GlassCard key={i} className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <img src={getOWMIconUrl(day.icon)} alt="" className="w-10 h-10" />
                  <div>
                    <p className="text-sm font-semibold text-white">{formatDay(day.date)}</p>
                    <p className="text-xs text-gray-500">{day.condition}</p>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-lg font-bold text-white">{day.tempMax}°</span>
                  <span className="text-sm text-gray-500 ml-2">{day.tempMin}°</span>
                </div>
              </div>
            </GlassCard>
          ))}
        </div>
      )}
    </motion.div>
  );
}
