/**
 * Map OWM icon codes to emoji and descriptions
 */
export const weatherIcons = {
  "01d": { emoji: "☀️", label: "Clear Sky" },
  "01n": { emoji: "🌙", label: "Clear Night" },
  "02d": { emoji: "⛅", label: "Few Clouds" },
  "02n": { emoji: "☁️", label: "Few Clouds" },
  "03d": { emoji: "☁️", label: "Scattered Clouds" },
  "03n": { emoji: "☁️", label: "Scattered Clouds" },
  "04d": { emoji: "☁️", label: "Overcast" },
  "04n": { emoji: "☁️", label: "Overcast" },
  "09d": { emoji: "🌧️", label: "Drizzle" },
  "09n": { emoji: "🌧️", label: "Drizzle" },
  "10d": { emoji: "🌦️", label: "Rain" },
  "10n": { emoji: "🌧️", label: "Rain" },
  "11d": { emoji: "⛈️", label: "Thunderstorm" },
  "11n": { emoji: "⛈️", label: "Thunderstorm" },
  "13d": { emoji: "❄️", label: "Snow" },
  "13n": { emoji: "❄️", label: "Snow" },
  "50d": { emoji: "🌫️", label: "Mist" },
  "50n": { emoji: "🌫️", label: "Mist" },
};

export function getWeatherEmoji(iconCode) {
  return weatherIcons[iconCode]?.emoji || "🌤️";
}

export function getWeatherLabel(iconCode) {
  return weatherIcons[iconCode]?.label || "Unknown";
}

/**
 * OWM icon URL
 */
export function getOWMIconUrl(iconCode, size = 2) {
  return `https://openweathermap.org/img/wn/${iconCode}@${size}x.png`;
}

/**
 * Format unix timestamp to readable time
 */
export function formatTime(unix) {
  if (!unix) return "--";
  return new Date(unix * 1000).toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}

/**
 * Format date string to day name
 */
export function formatDay(dateStr) {
  const d = new Date(dateStr);
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  if (d.toDateString() === today.toDateString()) return "Today";
  if (d.toDateString() === tomorrow.toDateString()) return "Tomorrow";

  return d.toLocaleDateString("en-IN", { weekday: "short", month: "short", day: "numeric" });
}

/**
 * Get greeting based on time of day
 */
export function getGreeting() {
  const h = new Date().getHours();
  if (h < 5) return "Good Night";
  if (h < 12) return "Good Morning";
  if (h < 17) return "Good Afternoon";
  if (h < 21) return "Good Evening";
  return "Good Night";
}

/**
 * Wind speed description
 */
export function windDescription(speed) {
  if (speed < 1) return "Calm";
  if (speed < 5) return "Light";
  if (speed < 11) return "Moderate";
  if (speed < 20) return "Strong";
  return "Very Strong";
}
