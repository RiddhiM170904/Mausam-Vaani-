import api from "./api";

// OpenWeatherMap free tier – replace with your own key
const OWM_KEY = import.meta.env.VITE_OWM_KEY || "";
const OWM_BASE = "https://api.openweathermap.org/data/2.5";
const OWM_GEO = "https://api.openweathermap.org/geo/1.0";

/**
 * Weather service — calls OpenWeatherMap directly for guest users
 * or proxies through our Node backend.
 */
export const weatherService = {
  /**
   * Full weather bundle: current + hourly + daily + alerts
   */
  async getFullWeather(lat, lon) {
    console.log(`Fetching weather for coordinates: ${lat}, ${lon}`);
    
    // Try backend first for logged in users
    try {
      const res = await api.get("/weather", { params: { lat, lon } });
      if (res.data) {
        console.log('Weather data from backend:', res.data);
        return res.data;
      }
    } catch (backendError) {
      console.warn('Backend weather fetch failed, trying direct API:', backendError.message);
      
      // Fallback to direct OWM call
      try {
        const owmData = await this.getFromOWM(lat, lon);
        console.log('Weather data from OpenWeatherMap direct:', owmData);
        return owmData;
      } catch (owmError) {
        console.warn('OpenWeatherMap direct fetch failed:', owmError.message);
        
        // Last resort: mock data with location info
        console.log('Using mock data as final fallback');
        return this.getMockData(lat, lon);
      }
    }
  },

  /**
   * Direct OpenWeatherMap OneCall-style fetch (free tier: current + forecast)
   */
  async getFromOWM(lat, lon) {
    const API_KEY = import.meta.env.VITE_OWM_KEY;
    
    if (!API_KEY) {
      console.warn('No OpenWeatherMap API key found. Using mock data.');
      return this.getMockData(lat, lon);
    }

    try {
      console.log(`Fetching from OpenWeatherMap with API key: ${API_KEY.substring(0, 8)}...`);
      
      const [currentRes, forecastRes] = await Promise.all([
        fetch(`${OWM_BASE}/weather?lat=${lat}&lon=${lon}&units=metric&appid=${API_KEY}`),
        fetch(`${OWM_BASE}/forecast?lat=${lat}&lon=${lon}&units=metric&appid=${API_KEY}`),
      ]);

      if (!currentRes.ok || !forecastRes.ok) {
        throw new Error(`API Error: ${currentRes.status} / ${forecastRes.status}`);
      }

      const current = await currentRes.json();
      const forecast = await forecastRes.json();

      console.log('OpenWeatherMap API Response - Current:', current);
      console.log('OpenWeatherMap API Response - Forecast:', forecast);

      return this.transformOWMData(current, forecast);
    } catch (error) {
      console.error('OpenWeatherMap API Error:', error);
      throw new Error(`Weather API failed: ${error.message}`);
    }
  },

  /**
   * Reverse geocode: lat/lon → city name
   */
  async reverseGeocode(lat, lon) {
    if (!OWM_KEY) return "New Delhi";
    try {
      const res = await fetch(
        `${OWM_GEO}/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${OWM_KEY}`
      );
      const data = await res.json();
      return data[0]?.name || "Unknown";
    } catch {
      return "Unknown";
    }
  },

  /**
   * Search city by name
   */
  async searchCity(query) {
    if (!OWM_KEY) return [];
    const res = await fetch(
      `${OWM_GEO}/direct?q=${encodeURIComponent(query)}&limit=5&appid=${OWM_KEY}`
    );
    return res.json();
  },

  /**
   * Transform raw OWM data to our app format
   */
  transformOWMData(current, forecast) {
    const hourly = (forecast.list || []).slice(0, 8).map((item) => ({
      time: item.dt_txt?.split(" ")[1]?.slice(0, 5) || "",
      temp: Math.round(item.main.temp),
      icon: item.weather[0]?.icon || "01d",
      condition: item.weather[0]?.main || "",
    }));

    const dailyMap = {};
    (forecast.list || []).forEach((item) => {
      const date = item.dt_txt?.split(" ")[0];
      if (!dailyMap[date]) {
        dailyMap[date] = {
          date,
          temps: [],
          icon: item.weather[0]?.icon || "01d",
          condition: item.weather[0]?.main || "",
        };
      }
      dailyMap[date].temps.push(item.main.temp);
    });

    const daily = Object.values(dailyMap)
      .slice(0, 7)
      .map((d) => ({
        date: d.date,
        tempMax: Math.round(Math.max(...d.temps)),
        tempMin: Math.round(Math.min(...d.temps)),
        icon: d.icon,
        condition: d.condition,
      }));

    return {
      city: current.name || "Unknown",
      current: {
        temp: Math.round(current.main?.temp || 0),
        feelsLike: Math.round(current.main?.feels_like || 0),
        humidity: current.main?.humidity || 0,
        wind: current.wind?.speed || 0,
        visibility: ((current.visibility || 10000) / 1000).toFixed(1),
        pressure: current.main?.pressure || 0,
        condition: current.weather?.[0]?.main || "Clear",
        description: current.weather?.[0]?.description || "clear sky",
        icon: current.weather?.[0]?.icon || "01d",
        uvi: 0,
        sunrise: current.sys?.sunrise || 0,
        sunset: current.sys?.sunset || 0,
      },
      hourly,
      daily,
      alerts: [],
    };
  },

  /**
   * Mock data when no API key is available — for development
   * Enhanced with location-based realistic data
   */
  async getMockData(lat, lon) {
    console.log(`Generating mock data for coordinates: ${lat}, ${lon}`);
    
    // Get city name for coordinates if possible
    let cityName = "Unknown Location";
    try {
      cityName = await this.reverseGeocode(lat, lon);
    } catch {
      // Use approximate city based on coordinates
      if (lat >= 28 && lat <= 29 && lon >= 76 && lon <= 78) {
        cityName = "New Delhi";
      } else if (lat >= 18 && lat <= 20 && lon >= 72 && lon <= 73) {
        cityName = "Mumbai";
      } else if (lat >= 12 && lat <= 13 && lon >= 77 && lon <= 78) {
        cityName = "Bangalore";
      } else if (lat >= 13 && lat <= 14 && lon >= 80 && lon <= 81) {
        cityName = "Chennai";
      } else {
        cityName = "Current Location";
      }
    }

    const now = new Date();
    const hours = Array.from({ length: 8 }, (_, i) => {
      const h = new Date(now.getTime() + i * 3600000);
      return {
        time: h.toTimeString().slice(0, 5),
        temp: Math.round(22 + Math.random() * 10),
        icon: ["01d", "02d", "03d", "10d", "01n"][Math.floor(Math.random() * 5)],
        condition: ["Clear", "Clouds", "Rain", "Haze"][Math.floor(Math.random() * 4)],
      };
    });

    const days = Array.from({ length: 7 }, (_, i) => {
      const d = new Date(now.getTime() + i * 86400000);
      return {
        date: d.toISOString().split("T")[0],
        tempMax: Math.round(28 + Math.random() * 8),
        tempMin: Math.round(18 + Math.random() * 5),
        icon: ["01d", "02d", "03d", "10d"][Math.floor(Math.random() * 4)],
        condition: ["Clear", "Clouds", "Rain", "Haze"][Math.floor(Math.random() * 4)],
      };
    });

    return {
      city: cityName,
      current: {
        temp: 28,
        feelsLike: 32,
        humidity: 65,
        wind: 3.5,
        visibility: "8.0",
        pressure: 1012,
        condition: "Partly Cloudy",
        description: "partly cloudy",
        icon: "02d",
        uvi: 6,
        sunrise: Math.floor(now.setHours(6, 15, 0) / 1000),
        sunset: Math.floor(now.setHours(18, 30, 0) / 1000),
      },
      hourly: hours,
      daily: days,
      alerts: [
        {
          event: "Demo Mode",
          description: `Showing sample weather data for ${cityName}. Add OpenWeatherMap API key for real data.`,
          severity: "info",
        },
      ],
    };
  },
};
