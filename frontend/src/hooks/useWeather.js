import { useState, useEffect, useCallback } from "react";
import { weatherService } from "../services/weatherService";

const REFRESH_INTERVAL = 10 * 60 * 1000; // 10 minutes

/**
 * Fetches and auto-refreshes weather data for given coordinates.
 */
export default function useWeather(lat, lon) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchWeather = useCallback(async () => {
    if (lat == null || lon == null) return;
    try {
      setLoading(true);
      const result = await weatherService.getFullWeather(lat, lon);
      setData(result);
      setError(null);
    } catch (err) {
      setError(err.message || "Failed to fetch weather");
    } finally {
      setLoading(false);
    }
  }, [lat, lon]);

  useEffect(() => {
    fetchWeather();
    const interval = setInterval(fetchWeather, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchWeather]);

  return { data, loading, error, refetch: fetchWeather };
}
