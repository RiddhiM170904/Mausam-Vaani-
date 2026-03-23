import { AI_CONFIG, hasWeatherKey } from './aiConfig.js';

const weatherCache = new Map();

function toNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function mapAqi(openWeatherAqi) {
  return ({ 1: 35, 2: 75, 3: 125, 4: 175, 5: 225 })[openWeatherAqi] || 80;
}

export async function fetchWeatherByCoordinates(lat, lon) {
  const nLat = toNumber(lat, null);
  const nLon = toNumber(lon, null);

  if (nLat === null || nLon === null) {
    throw new Error('Valid lat/lon required');
  }

  if (!hasWeatherKey()) {
    throw new Error('VITE_OWM_KEY missing in frontend environment');
  }

  const cacheKey = `${nLat.toFixed(3)}:${nLon.toFixed(3)}`;
  const cached = weatherCache.get(cacheKey);
  if (cached && Date.now() - cached.ts < AI_CONFIG.weatherTtlMs) {
    return cached.value;
  }

  const key = AI_CONFIG.openWeatherKey;
  const weatherUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${nLat}&lon=${nLon}&units=metric&appid=${key}`;
  const forecastUrl = `https://api.openweathermap.org/data/2.5/forecast?lat=${nLat}&lon=${nLon}&units=metric&cnt=3&appid=${key}`;
  const aqiUrl = `https://api.openweathermap.org/data/2.5/air_pollution?lat=${nLat}&lon=${nLon}&appid=${key}`;

  const [weatherRes, forecastRes, aqiRes] = await Promise.all([
    fetch(weatherUrl),
    fetch(forecastUrl),
    fetch(aqiUrl),
  ]);

  if (!weatherRes.ok || !forecastRes.ok || !aqiRes.ok) {
    throw new Error('Failed to fetch weather provider responses');
  }

  const [weather, forecast, aqi] = await Promise.all([
    weatherRes.json(),
    forecastRes.json(),
    aqiRes.json(),
  ]);

  const normalized = {
    temp: toNumber(weather?.main?.temp, 0),
    rain_probability: Math.max(0, Math.min(1, toNumber(forecast?.list?.[0]?.pop, 0))),
    aqi: mapAqi(aqi?.list?.[0]?.main?.aqi),
    wind_speed: toNumber(weather?.wind?.speed, 0),
    humidity: toNumber(weather?.main?.humidity, 0),
    weather_main: weather?.weather?.[0]?.main || 'Clear',
    weather_description: weather?.weather?.[0]?.description || 'clear sky',
    city: weather?.name || '',
    country: weather?.sys?.country || '',
    observed_at: new Date().toISOString(),
  };

  weatherCache.set(cacheKey, { ts: Date.now(), value: normalized });
  return normalized;
}
