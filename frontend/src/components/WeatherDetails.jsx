import GlassCard from "./GlassCard";
import { formatTime, formatVisibilityKm, formatWindKmh, windDescription } from "../utils/helpers";
import {
  WiHumidity,
  WiStrongWind,
  WiBarometer,
  WiSunrise,
  WiSunset,
  WiDaySunny,
} from "react-icons/wi";
import { HiOutlineEye } from "react-icons/hi2";

const aqiLabel = {
  1: "Good",
  2: "Fair",
  3: "Moderate",
  4: "Poor",
  5: "Very Poor",
  6: "Hazardous",
};

/**
 * Grid of weather detail metrics.
 */
export default function WeatherDetails({ data }) {
  if (!data) return null;

  const items = [
    {
      icon: <WiHumidity size={28} className="text-blue-400" />,
      label: "Humidity",
      value: `${data.humidity}%`,
    },
    {
      icon: <WiStrongWind size={28} className="text-teal-400" />,
      label: "Wind",
      value: formatWindKmh(data.wind),
      sub: windDescription(data.wind),
    },
    {
      icon: <HiOutlineEye size={22} className="text-gray-400" />,
      label: "Visibility",
      value: formatVisibilityKm(data.visibility),
    },
    {
      icon: <WiBarometer size={28} className="text-purple-400" />,
      label: "Pressure",
      value: `${data.pressure} hPa`,
    },
    {
      icon: <WiSunrise size={28} className="text-orange-400" />,
      label: "Sunrise",
      value: formatTime(data.sunrise),
    },
    {
      icon: <WiSunset size={28} className="text-rose-400" />,
      label: "Sunset",
      value: formatTime(data.sunset),
    },
  ];

  if (data.uvi !== undefined && data.uvi > 0) {
    items.splice(2, 0, {
      icon: <WiDaySunny size={28} className="text-yellow-400" />,
      label: "UV Index",
      value: data.uvi,
    });
  }

  if (data.aqi) {
    items.splice(3, 0, {
      icon: <span className="inline-block h-3 w-3 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.75)]" />,
      label: "AQI",
      value: `${data.aqi} (${aqiLabel[data.aqi] || "Unknown"})`,
    });
  }

  return (
    <div className="grid grid-cols-1 gap-4 px-1 sm:grid-cols-2 sm:px-0 lg:grid-cols-3 xl:grid-cols-4">
      {items.map((item) => (
        <GlassCard key={item.label} className="p-4 flex flex-col gap-2" hover>
          <div className="flex items-center gap-2">
            {item.icon}
            <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">
              {item.label}
            </span>
          </div>
          <p className="text-lg font-semibold text-white">{item.value}</p>
          {item.sub && <p className="text-xs text-gray-500">{item.sub}</p>}
        </GlassCard>
      ))}
    </div>
  );
}
