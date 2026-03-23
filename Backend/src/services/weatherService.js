const { cache } = require('./cacheService');

const WEATHER_TTL_MS = 10 * 60 * 1000;

function keyFromCoordinates(lat, lon) {
  return `weather:${lat.toFixed(3)}:${lon.toFixed(3)}`;
}

function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function mapAqi(openWeatherAqi) {
  const map = {
    1: 35,
    2: 75,
    3: 125,
    4: 175,
    5: 225,
  };
  return map[openWeatherAqi] || 80;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`External API error ${response.status}: ${text}`);
  }

  return response.json();
}

async function getCurrentWeather(lat, lon) {
  const safeLat = toNumber(lat, null);
  const safeLon = toNumber(lon, null);

  if (safeLat === null || safeLon === null) {
    throw new Error('Valid latitude and longitude are required');
  }

  const cacheKey = keyFromCoordinates(safeLat, safeLon);
  const cached = cache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const apiKey = process.env.OPENWEATHER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENWEATHER_API_KEY is missing in environment');
  }

  const weatherUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${safeLat}&lon=${safeLon}&units=metric&appid=${apiKey}`;
  const forecastUrl = `https://api.openweathermap.org/data/2.5/forecast?lat=${safeLat}&lon=${safeLon}&units=metric&cnt=3&appid=${apiKey}`;
  const aqiUrl = `https://api.openweathermap.org/data/2.5/air_pollution?lat=${safeLat}&lon=${safeLon}&appid=${apiKey}`;

  const [current, forecast, aqi] = await Promise.all([
    fetchJson(weatherUrl),
    fetchJson(forecastUrl),
    fetchJson(aqiUrl),
  ]);

  const rainProbability = toNumber(forecast?.list?.[0]?.pop, 0);
  const normalized = {
    temp: toNumber(current?.main?.temp, 0),
    rain_probability: rainProbability,
    aqi: mapAqi(aqi?.list?.[0]?.main?.aqi),
    wind_speed: toNumber(current?.wind?.speed, 0),
    humidity: toNumber(current?.main?.humidity, 0),
    weather_main: current?.weather?.[0]?.main || 'Clear',
    weather_description: current?.weather?.[0]?.description || 'clear sky',
    city: current?.name || '',
    country: current?.sys?.country || '',
    observed_at: new Date().toISOString(),
  };

  cache.set(cacheKey, normalized, WEATHER_TTL_MS);
  return normalized;
}

module.exports = {
  getCurrentWeather,
};
