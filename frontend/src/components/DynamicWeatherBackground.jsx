import { motion } from "framer-motion";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";

const isNightNow = () => {
  const hour = new Date().getHours();
  return hour < 6 || hour >= 18;
};

const pickTheme = (weather) => {
  const condition = String(weather?.current?.condition || "").toLowerCase();
  const icon = String(weather?.current?.icon || "");
  const night = icon.endsWith("n") || isNightNow();

  if (condition.includes("rain") || condition.includes("drizzle") || condition.includes("thunder")) {
    return {
      base: night ? "from-slate-950 via-slate-900 to-indigo-950" : "from-slate-900 via-sky-900 to-indigo-900",
      glowA: "bg-cyan-500/20",
      glowB: "bg-indigo-500/25",
      rain: true,
      stars: false,
    };
  }

  if (condition.includes("cloud") || condition.includes("mist") || condition.includes("haze") || condition.includes("fog")) {
    return {
      base: night ? "from-slate-950 via-slate-900 to-zinc-900" : "from-slate-900 via-gray-800 to-zinc-900",
      glowA: "bg-slate-400/15",
      glowB: "bg-sky-400/10",
      rain: false,
      stars: night,
    };
  }

  if (night) {
    return {
      base: "from-slate-950 via-indigo-950 to-blue-950",
      glowA: "bg-indigo-500/20",
      glowB: "bg-blue-500/15",
      rain: false,
      stars: true,
    };
  }

  return {
    base: "from-sky-950 via-blue-900 to-cyan-900",
    glowA: "bg-cyan-400/20",
    glowB: "bg-blue-500/20",
    rain: false,
    stars: false,
  };
};

export default function DynamicWeatherBackground() {
  const { location } = useLocation();
  const { data } = useWeather(location?.lat, location?.lon);

  const theme = pickTheme(data);

  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className={`absolute inset-0 bg-linear-to-br ${theme.base}`} />

      <motion.div
        className={`absolute -left-24 top-12 h-72 w-72 rounded-full blur-3xl ${theme.glowA}`}
        animate={{ x: [0, 40, 0], y: [0, 20, 0] }}
        transition={{ duration: 16, repeat: Infinity, ease: "easeInOut" }}
      />

      <motion.div
        className={`absolute -right-20 bottom-10 h-80 w-80 rounded-full blur-3xl ${theme.glowB}`}
        animate={{ x: [0, -50, 0], y: [0, -25, 0] }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
      />

      {theme.rain && (
        <div className="absolute inset-0 opacity-35">
          {Array.from({ length: 26 }).map((_, i) => (
            <motion.span
              key={i}
              className="absolute top-[-20%] block h-14 w-px rounded-full bg-cyan-200/70"
              style={{ left: `${(i * 4) % 100}%` }}
              animate={{ y: [0, 900], x: [0, -30] }}
              transition={{
                duration: 1.3 + (i % 5) * 0.25,
                repeat: Infinity,
                ease: "linear",
                delay: (i % 7) * 0.2,
              }}
            />
          ))}
        </div>
      )}

      {theme.stars && (
        <div className="absolute inset-0 opacity-60">
          {Array.from({ length: 32 }).map((_, i) => (
            <motion.span
              key={i}
              className="absolute h-1 w-1 rounded-full bg-white"
              style={{ left: `${(i * 13) % 100}%`, top: `${(i * 19) % 100}%` }}
              animate={{ opacity: [0.2, 1, 0.2] }}
              transition={{ duration: 2.5 + (i % 4), repeat: Infinity, delay: i * 0.07 }}
            />
          ))}
        </div>
      )}

      <div className="absolute inset-0 bg-black/30" />
    </div>
  );
}
