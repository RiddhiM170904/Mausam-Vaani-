import { useState } from "react";
import { motion } from "framer-motion";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import Loader from "../components/Loader";
import GlassCard from "../components/GlassCard";
import {
  getOWMIconUrl,
  formatDay,
  formatTime,
  formatHourLabel,
  formatTemperatureC,
  formatWindKmh,
  formatVisibilityKm,
} from "../utils/helpers";
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

  const summaryItems = [
    { label: "Feels Like", value: formatTemperatureC(data?.current?.feelsLike), tone: "text-indigo-300" },
    { label: "Humidity", value: `${data?.current?.humidity ?? "--"}%`, tone: "text-blue-300" },
    { label: "Wind", value: formatWindKmh(data?.current?.wind), tone: "text-cyan-300" },
    { label: "Visibility", value: formatVisibilityKm(data?.current?.visibility), tone: "text-emerald-300" },
    { label: "Pressure", value: `${data?.current?.pressure ?? "--"} hPa`, tone: "text-violet-300" },
    { label: "Sunrise", value: formatTime(data?.current?.sunrise), tone: "text-amber-300" },
    { label: "Sunset", value: formatTime(data?.current?.sunset), tone: "text-rose-300" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Forecast</h1>
        <div className="flex gap-1 p-1 rounded-xl bg-white/6">
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

      {/* Weather summary */}
      <GlassCard className="p-4 sm:p-5" hover={false}>
        <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-white/8">
          <div>
            <p className="text-xs tracking-wider text-gray-500 uppercase">Current Snapshot</p>
            <p className="text-lg font-semibold text-white">{data?.city || "Current Location"}</p>
            <p className="text-sm text-gray-400 capitalize">{data?.current?.description || data?.current?.condition}</p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-light tracking-tight text-white">{formatTemperatureC(data?.current?.temp)}</p>
            <p className="text-xs text-gray-500">Updated from latest forecast data</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-4 sm:grid-cols-3 lg:grid-cols-7">
          {summaryItems.map((item) => (
            <div key={item.label} className="rounded-xl bg-white/3 px-3 py-2.5 border border-white/6">
              <p className="text-[10px] uppercase tracking-wider text-gray-500">{item.label}</p>
              <p className={`mt-1 text-sm font-semibold ${item.tone}`}>{item.value}</p>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Chart */}
      <GlassCard className="p-4 sm:p-6" hover={false}>
        <h3 className="mb-4 text-sm font-semibold text-gray-400">
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
                  tickFormatter={(value) => formatHourLabel(value)}
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
                  formatter={(value) => [formatTemperatureC(value), "Temperature"]}
                  labelFormatter={(label) => formatHourLabel(label)}
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
                  formatter={(value, name) => [formatTemperatureC(value), name === "max" ? "High" : "Low"]}
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
        <div className="grid grid-cols-1 gap-4 px-1 sm:grid-cols-2 sm:px-0 lg:grid-cols-4">
          {(data.hourly || []).map((h, i) => (
            <GlassCard key={i} className="flex flex-col items-center gap-2 p-4">
              <span className="text-xs text-gray-400">{formatHourLabel(h.time)}</span>
              <img src={getOWMIconUrl(h.icon)} alt="" className="w-10 h-10" />
              <span className="text-lg font-bold text-white">{formatTemperatureC(h.temp)}</span>
              <span className="text-xs text-gray-500">{h.condition}</span>
              <span className="text-xs text-blue-300">Rain chance: {h.rainProbability ?? h.pop ?? 0}%</span>
            </GlassCard>
          ))}
        </div>
      ) : (
        <div className="px-1 space-y-2 sm:px-0">
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
                  <span className="text-lg font-bold text-white">{formatTemperatureC(day.tempMax)}</span>
                  <span className="ml-2 text-sm text-gray-500">{formatTemperatureC(day.tempMin)}</span>
                  <p className="mt-1 text-xs text-gray-500">
                    Range: {Math.max(0, (day.tempMax ?? 0) - (day.tempMin ?? 0))}\u00B0C
                  </p>
                </div>
              </div>
            </GlassCard>
          ))}
        </div>
      )}
    </motion.div>
  );
}
