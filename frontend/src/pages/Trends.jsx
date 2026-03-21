import { motion } from "framer-motion";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid } from "recharts";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import GlassCard from "../components/GlassCard";
import Loader from "../components/Loader";

function safeNumber(value, fallback = 0) {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

export default function Trends() {
  const { location } = useLocation();
  const { data, loading } = useWeather(location?.lat, location?.lon);

  if (loading) return <Loader text="Loading analytics..." />;
  if (!data) return null;

  const hourly = (data.hourly || []).slice(0, 12);

  const tempData = hourly.map((h) => ({
    time: h.time,
    temp: safeNumber(h.temp),
  }));

  const rainData = hourly.map((h) => ({
    time: h.time,
    rain: safeNumber(h.pop ?? h.rainProbability ?? 0),
  }));

  const aqiData = hourly
    .map((h) => ({
      time: h.time,
      aqi: h.aqi == null ? null : safeNumber(h.aqi),
    }))
    .filter((item) => item.aqi != null);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mx-auto w-full max-w-5xl space-y-4 sm:space-y-6">
      <div className="space-y-1 px-1">
        <h1 className="text-2xl font-bold text-white">Trends and Analytics</h1>
        <p className="text-sm text-gray-400">Track weather patterns for better planning decisions.</p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <GlassCard className="p-4" hover={false}>
          <p className="text-xs uppercase tracking-wide text-gray-500">Current Temperature</p>
          <p className="mt-2 text-2xl font-bold text-white">{data?.current?.temp ?? "--"}°C</p>
        </GlassCard>
        <GlassCard className="p-4" hover={false}>
          <p className="text-xs uppercase tracking-wide text-gray-500">Humidity</p>
          <p className="mt-2 text-2xl font-bold text-white">{data?.current?.humidity ?? "--"}%</p>
        </GlassCard>
        <GlassCard className="p-4" hover={false}>
          <p className="text-xs uppercase tracking-wide text-gray-500">Wind Speed</p>
          <p className="mt-2 text-2xl font-bold text-white">{data?.current?.wind ?? "--"} km/h</p>
        </GlassCard>
      </div>

      <GlassCard className="p-4 sm:p-5" hover={false}>
        <h2 className="mb-3 text-sm font-semibold text-gray-300">Temperature Trend</h2>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={tempData}>
              <defs>
                <linearGradient id="tempTrendGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#818cf8" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
              <Tooltip />
              <Area type="monotone" dataKey="temp" stroke="#818cf8" strokeWidth={2.5} fill="url(#tempTrendGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <GlassCard className="p-4 sm:p-5" hover={false}>
          <h2 className="mb-3 text-sm font-semibold text-gray-300">Rain Probability Trend</h2>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rainData}>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                <XAxis dataKey="time" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
                <Tooltip />
                <Line type="monotone" dataKey="rain" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>

        <GlassCard className="p-4 sm:p-5" hover={false}>
          <h2 className="mb-3 text-sm font-semibold text-gray-300">AQI Trend</h2>
          {aqiData.length ? (
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={aqiData}>
                  <CartesianGrid stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
                  <Tooltip />
                  <Line type="monotone" dataKey="aqi" stroke="#f97316" strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-sm text-gray-500">AQI trend data is not available for this location right now.</p>
          )}
        </GlassCard>
      </div>
    </motion.div>
  );
}
