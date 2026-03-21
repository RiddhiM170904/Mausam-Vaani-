import { useState, useEffect, useCallback, useRef } from "react";
import { weatherService } from "../services/weatherService";

const REFRESH_INTERVAL = 10 * 60 * 1000; // 10 minutes
const WEATHER_DEBUG_PREFIX = "[WeatherDebug]";

/**
 * Fetches and auto-refreshes weather data for given coordinates.
 */
export default function useWeather(lat, lon) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const inFlightRequestRef = useRef(null);
  const missingCoordsLoggedRef = useRef(false);

  const fetchWeather = useCallback(async ({ force = false } = {}) => {
    if (lat == null || lon == null) {
      if (!missingCoordsLoggedRef.current) {
        console.info(`${WEATHER_DEBUG_PREFIX} Missing coordinates`, { lat, lon });
        missingCoordsLoggedRef.current = true;
      }
      return;
    }

    missingCoordsLoggedRef.current = false;

    const requestKey = `${lat},${lon}`;
    if (inFlightRequestRef.current === requestKey) {
      console.info(`${WEATHER_DEBUG_PREFIX} Skip duplicate in-flight fetch`, { lat, lon });
      return;
    }

    console.info(`${WEATHER_DEBUG_PREFIX} Fetch start`, { lat, lon });

    try {
      inFlightRequestRef.current = requestKey;
      setLoading(true);
      const result = await weatherService.getFullWeatherCached(lat, lon, { force });
      setData(result);
      setError(null);
      console.info(`${WEATHER_DEBUG_PREFIX} Fetch success`, {
        city: result?.city,
        hasCurrent: Boolean(result?.current),
        hourlyCount: result?.hourly?.length || 0,
        dailyCount: result?.daily?.length || 0,
        alertsCount: result?.alerts?.length || 0,
      });
    } catch (err) {
      const resolvedMessage = err?.message || "Failed to fetch weather";
      setError(resolvedMessage);
      console.error(`${WEATHER_DEBUG_PREFIX} Fetch failed`, {
        message: resolvedMessage,
        lat,
        lon,
        stack: err?.stack,
      });
    } finally {
      if (inFlightRequestRef.current === requestKey) {
        inFlightRequestRef.current = null;
      }
      setLoading(false);
    }
  }, [lat, lon]);

  useEffect(() => {
    fetchWeather();
    const interval = setInterval(fetchWeather, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchWeather]);

  return {
    data,
    loading,
    error,
    refetch: () => fetchWeather({ force: true }),
  };
}
