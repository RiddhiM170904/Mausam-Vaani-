const WEATHER_API_KEY = import.meta.env.VITE_WEATHERAPI_KEY || "";
const WEATHER_API_BASE = "https://api.weatherapi.com/v1";
const OWM_KEY = import.meta.env.VITE_OWM_KEY || "";
const OWM_BASE = "https://api.openweathermap.org/data/2.5";
const OWM_GEO = "https://api.openweathermap.org/geo/1.0";
const GOOGLE_PLACES_API_KEY = import.meta.env.VITE_GOOGLE_PLACES_API_KEY || "";
const WEATHER_PROVIDER = (import.meta.env.VITE_WEATHER_PROVIDER || "auto").toLowerCase();
const WEATHER_DEBUG_PREFIX = "[WeatherDebug]";
const WEATHER_CACHE_TTL_MS = 5 * 60 * 1000;
const weatherResponseCache = new Map();
const weatherInFlightRequests = new Map();
const PLACES_CACHE_TTL_MS = 60 * 1000;
const placesResponseCache = new Map();
const placesInFlightRequests = new Map();
const OSM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse";
let hasLoggedPlaces403 = false;

const getCoordCacheKey = (lat, lon) => `${Number(lat).toFixed(4)},${Number(lon).toFixed(4)}`;

/**
  * Weather service — calls OpenWeatherMap directly.
 */
export const weatherService = {
  async getOpenStreetHyperlocalPlace(lat, lon) {
    try {
      const response = await fetch(
        `${OSM_REVERSE_URL}?format=jsonv2&lat=${Number(lat)}&lon=${Number(lon)}&zoom=18&addressdetails=1&namedetails=1`,
        {
          headers: {
            "Accept-Language": "en",
          },
        }
      );

      if (!response.ok) {
        return null;
      }

      const payload = await response.json();
      const address = payload?.address || {};

      const name =
        payload?.name ||
        payload?.namedetails?.name ||
        address?.amenity ||
        address?.building ||
        address?.tourism ||
        address?.leisure ||
        address?.attraction ||
        address?.road ||
        address?.neighbourhood ||
        address?.suburb ||
        null;

      if (!name) {
        return null;
      }

      return {
        placeId: payload?.place_id ? String(payload.place_id) : null,
        name,
        formattedAddress: payload?.display_name || name,
        coordinates: {
          lat: Number(payload?.lat ?? lat),
          lon: Number(payload?.lon ?? lon),
        },
      };
    } catch {
      return null;
    }
  },

  async getFullWeatherCached(lat, lon, { force = false } = {}) {
    const key = getCoordCacheKey(lat, lon);
    const now = Date.now();
    const cached = weatherResponseCache.get(key);

    if (!force && cached && now - cached.timestamp < WEATHER_CACHE_TTL_MS) {
      console.info(`${WEATHER_DEBUG_PREFIX} Cache hit`, {
        lat,
        lon,
        ageMs: now - cached.timestamp,
      });
      return cached.data;
    }

    const inFlight = weatherInFlightRequests.get(key);
    if (!force && inFlight) {
      console.info(`${WEATHER_DEBUG_PREFIX} Reusing in-flight request`, { lat, lon });
      return inFlight;
    }

    const requestPromise = this.getFullWeather(lat, lon)
      .then((data) => {
        weatherResponseCache.set(key, { timestamp: Date.now(), data });
        return data;
      })
      .finally(() => {
        weatherInFlightRequests.delete(key);
      });

    weatherInFlightRequests.set(key, requestPromise);
    return requestPromise;
  },

  /**
   * Full weather bundle: current + hourly + daily + alerts
   */
  async getFullWeather(lat, lon) {
    const selectedProvider =
      WEATHER_PROVIDER === "weatherapi"
        ? "weatherapi"
        : WEATHER_PROVIDER === "owm"
          ? "owm"
          : WEATHER_API_KEY
            ? "weatherapi"
            : OWM_KEY
              ? "owm"
              : "none";

    console.info(`${WEATHER_DEBUG_PREFIX} getFullWeather called`, {
      lat,
      lon,
      provider: selectedProvider,
      hasWeatherApiKey: Boolean(WEATHER_API_KEY),
      hasOwmKey: Boolean(OWM_KEY),
      providerMode: WEATHER_PROVIDER,
    });

    if (selectedProvider === "weatherapi") {
      if (!WEATHER_API_KEY) {
        throw new Error("VITE_WEATHER_PROVIDER is weatherapi, but VITE_WEATHERAPI_KEY is missing.");
      }

      try {
        return await this.getRealtimeWeather(lat, lon);
      } catch (err) {
        const canFallbackToOwm =
          OWM_KEY &&
          /status\s*(401|403)/i.test(err?.message || "");

        if (canFallbackToOwm) {
          console.warn(`${WEATHER_DEBUG_PREFIX} WeatherAPI auth failed, falling back to OWM`);
          return this.getRealtimeWeatherFromOWM(lat, lon);
        }

        throw err;
      }
    }

    if (selectedProvider === "owm") {
      if (!OWM_KEY) {
        throw new Error("VITE_WEATHER_PROVIDER is owm, but VITE_OWM_KEY is missing.");
      }
      return this.getRealtimeWeatherFromOWM(lat, lon);
    }

    throw new Error("Missing weather API key. Set VITE_OWM_KEY (or VITE_WEATHERAPI_KEY).");
  },

  async getRealtimeWeather(lat, lon) {
    const url = `${WEATHER_API_BASE}/forecast.json?key=${WEATHER_API_KEY}&q=${lat},${lon}&days=7&aqi=yes&alerts=yes`;
    console.info(`${WEATHER_DEBUG_PREFIX} WeatherAPI request`, {
      endpoint: "/forecast.json",
      lat,
      lon,
    });

    const response = await fetch(url);

    if (!response.ok) {
      const errorBody = await response.text().catch(() => "");
      console.error(`${WEATHER_DEBUG_PREFIX} WeatherAPI failed`, {
        status: response.status,
        body: errorBody?.slice?.(0, 200),
      });
      throw new Error(`Weather API failed with status ${response.status}`);
    }

    const payload = await response.json();
    return this.transformWeatherApiData(payload);
  },

  async getRealtimeWeatherFromOWM(lat, lon) {
    // Free-tier OWM path: single request that includes upcoming hourly and daily-capable data.
    const url = `${OWM_BASE}/forecast?lat=${lat}&lon=${lon}&units=metric&appid=${OWM_KEY}`;
    console.info(`${WEATHER_DEBUG_PREFIX} OWM request`, {
      endpoint: "/forecast",
      lat,
      lon,
    });

    const response = await fetch(url);

    if (!response.ok) {
      const errorBody = await response.text().catch(() => "");
      console.error(`${WEATHER_DEBUG_PREFIX} OWM failed`, {
        status: response.status,
        body: errorBody?.slice?.(0, 200),
      });
      throw new Error(`OpenWeather forecast failed with status ${response.status}`);
    }

    const payload = await response.json();
    return this.transformOWMForecastData(payload);
  },

  /**
   * Reverse geocode: lat/lon → city name
   */
  async reverseGeocode(lat, lon) {
    const preferOwm = WEATHER_PROVIDER === "owm";
    const preferWeatherApi = WEATHER_PROVIDER === "weatherapi";

    if (!preferOwm && WEATHER_API_KEY) {
      try {
        const res = await fetch(
          `${WEATHER_API_BASE}/current.json?key=${WEATHER_API_KEY}&q=${lat},${lon}&aqi=no`
        );
        if (!res.ok) return "Unknown";
        const data = await res.json();
        return data?.location?.name || "Unknown";
      } catch {
        return "Unknown";
      }
    }

    if (OWM_KEY) {
      try {
        const res = await fetch(
          `${OWM_GEO}/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${OWM_KEY}`
        );
        if (!res.ok) return "Unknown";
        const data = await res.json();
        return data?.[0]?.name || "Unknown";
      } catch {
        return "Unknown";
      }
    }

    if (preferWeatherApi && WEATHER_API_KEY) {
      try {
        const res = await fetch(
          `${WEATHER_API_BASE}/current.json?key=${WEATHER_API_KEY}&q=${lat},${lon}&aqi=no`
        );
        if (!res.ok) return "Unknown";
        const data = await res.json();
        return data?.location?.name || "Unknown";
      } catch {
        return "Unknown";
      }
    }

    return "Unknown";
  },

  /**
   * Hyperlocal place lookup using Google Places API.
   * Returns nearest place/address within the provided radius (default: 100m).
   */
  async getNearbyPlaceName(lat, lon, options = {}) {
    const radius = Math.min(Math.max(Number(options?.radius) || 100, 1), 50000);
    const key = `${Number(lat).toFixed(5)},${Number(lon).toFixed(5)},${radius}`;
    const cached = placesResponseCache.get(key);
    const now = Date.now();

    if (cached && now - cached.timestamp < PLACES_CACHE_TTL_MS) {
      return cached.data;
    }

    const existing = placesInFlightRequests.get(key);
    if (existing) {
      return existing;
    }

    if (!GOOGLE_PLACES_API_KEY) {
      const osmFallback = await this.getOpenStreetHyperlocalPlace(lat, lon);
      placesResponseCache.set(key, { timestamp: Date.now(), data: osmFallback });
      return osmFallback;
    }

    const requestPromise = (async () => {
      const response = await fetch("https://places.googleapis.com/v1/places:searchNearby", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
          "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location",
        },
        body: JSON.stringify({
          maxResultCount: 1,
          rankPreference: "DISTANCE",
          locationRestriction: {
            circle: {
              center: {
                latitude: Number(lat),
                longitude: Number(lon),
              },
              radius,
            },
          },
        }),
      });

      if (!response.ok) {
        const errorBody = await response.text().catch(() => "");
        if (response.status === 403 && !hasLoggedPlaces403) {
          hasLoggedPlaces403 = true;
          console.warn(`${WEATHER_DEBUG_PREFIX} Places lookup failed (403). Enable Places API (New), attach billing, and allow localhost referrer for this key.`);
        } else if (response.status !== 403) {
          console.warn(`${WEATHER_DEBUG_PREFIX} Places lookup failed`, {
            status: response.status,
            body: errorBody?.slice?.(0, 150),
          });
        }
        const osmFallback = await this.getOpenStreetHyperlocalPlace(lat, lon);
        placesResponseCache.set(key, { timestamp: Date.now(), data: osmFallback });
        return osmFallback;
      }

      const payload = await response.json();
      const place = payload?.places?.[0];

      if (!place) {
        const osmFallback = await this.getOpenStreetHyperlocalPlace(lat, lon);
        placesResponseCache.set(key, { timestamp: Date.now(), data: osmFallback });
        return osmFallback;
      }

      const name = place?.displayName?.text || place?.formattedAddress || null;
      if (!name) {
        const osmFallback = await this.getOpenStreetHyperlocalPlace(lat, lon);
        placesResponseCache.set(key, { timestamp: Date.now(), data: osmFallback });
        return osmFallback;
      }

      const result = {
        placeId: place.id || null,
        name,
        formattedAddress: place.formattedAddress || name,
        coordinates: {
          lat: place?.location?.latitude ?? Number(lat),
          lon: place?.location?.longitude ?? Number(lon),
        },
      };

      placesResponseCache.set(key, { timestamp: Date.now(), data: result });
      return result;
    })().catch(async (err) => {
      console.warn(`${WEATHER_DEBUG_PREFIX} Places lookup error`, err?.message || err);
      return this.getOpenStreetHyperlocalPlace(lat, lon);
    }).finally(() => {
      placesInFlightRequests.delete(key);
    });

    placesInFlightRequests.set(key, requestPromise);
    return requestPromise;
  },

  /**
   * Search city by name
   */
  async searchCity(query) {
    const preferOwm = WEATHER_PROVIDER === "owm";
    const preferWeatherApi = WEATHER_PROVIDER === "weatherapi";

    if (preferOwm && OWM_KEY) {
      const res = await fetch(
        `${OWM_GEO}/direct?q=${encodeURIComponent(query)}&limit=5&appid=${OWM_KEY}`
      );
      if (!res.ok) return [];
      const result = await res.json();
      return (result || []).map((item) => ({
        name: item.name,
        state: item.state,
        country: item.country,
        lat: item.lat,
        lon: item.lon,
      }));
    }

    if (!preferOwm && WEATHER_API_KEY) {
      const res = await fetch(
        `${WEATHER_API_BASE}/search.json?key=${WEATHER_API_KEY}&q=${encodeURIComponent(query)}`
      );

      if (res.ok) {
        const result = await res.json();
        return (result || []).map((item) => ({
          name: item.name,
          state: item.region,
          country: item.country,
          lat: item.lat,
          lon: item.lon,
        }));
      }
    }

    if (OWM_KEY) {
      const res = await fetch(
        `${OWM_GEO}/direct?q=${encodeURIComponent(query)}&limit=5&appid=${OWM_KEY}`
      );
      if (!res.ok) return [];
      const result = await res.json();
      return (result || []).map((item) => ({
        name: item.name,
        state: item.state,
        country: item.country,
        lat: item.lat,
        lon: item.lon,
      }));
    }

    if (preferWeatherApi && WEATHER_API_KEY) {
      const res = await fetch(
        `${WEATHER_API_BASE}/search.json?key=${WEATHER_API_KEY}&q=${encodeURIComponent(query)}`
      );
      if (!res.ok) return [];
      const result = await res.json();
      return (result || []).map((item) => ({
        name: item.name,
        state: item.region,
        country: item.country,
        lat: item.lat,
        lon: item.lon,
      }));
    }

    return [];
  },

  transformWeatherApiData(payload) {
    const location = payload?.location || {};
    const current = payload?.current || {};
    const forecastDays = payload?.forecast?.forecastday || [];
    const alerts = payload?.alerts?.alert || [];

    const localDate = location?.localtime ? location.localtime.split(" ")[0] : null;
    const firstDay = forecastDays[0] || null;
    const firstDayDate = firstDay?.date || localDate;

    const toUnix = (timeLabel) => {
      if (!timeLabel || !firstDayDate) return 0;
      const dt = new Date(`${firstDayDate} ${timeLabel}`);
      if (Number.isNaN(dt.getTime())) return 0;
      return Math.floor(dt.getTime() / 1000);
    };

    const flatHours = forecastDays
      .flatMap((d) => d.hour || [])
      .slice(0, 24)
      .map((h) => ({
        time: h.time?.split(" ")[1]?.slice(0, 5) || "",
        temp: Math.round(h.temp_c ?? 0),
        icon: h.condition?.icon ? `https:${h.condition.icon}` : "",
        condition: h.condition?.text || "",
        pop: Math.round(h.chance_of_rain ?? 0),
        rainProbability: Math.round(h.chance_of_rain ?? 0),
        aqi: h.air_quality?.["us-epa-index"] || null,
      }));

    const daily = forecastDays.slice(0, 7).map((d) => ({
      date: d.date,
      tempMax: Math.round(d.day?.maxtemp_c ?? 0),
      tempMin: Math.round(d.day?.mintemp_c ?? 0),
      icon: d.day?.condition?.icon ? `https:${d.day.condition.icon}` : "",
      condition: d.day?.condition?.text || "",
    }));

    return {
      city: location?.name || "Unknown",
      current: {
        temp: Math.round(current?.temp_c ?? 0),
        feelsLike: Math.round(current?.feelslike_c ?? 0),
        humidity: current?.humidity ?? 0,
        wind: current?.wind_kph ?? 0,
        visibility: Number(current?.vis_km ?? 0).toFixed(1),
        pressure: current?.pressure_mb ?? 0,
        condition: current?.condition?.text || "Clear",
        description: current?.condition?.text?.toLowerCase() || "clear",
        icon: current?.condition?.icon ? `https:${current.condition.icon}` : "",
        uvi: current?.uv ?? 0,
        aqi: current?.air_quality?.["us-epa-index"] || null,
        sunrise: toUnix(firstDay?.astro?.sunrise),
        sunset: toUnix(firstDay?.astro?.sunset),
      },
      hourly: flatHours,
      daily,
      alerts: alerts.map((a) => ({
        event: a?.headline || "Weather Alert",
        description: a?.desc || a?.instruction || "Please stay alert for changing weather.",
        severity: (a?.severity || "warning").toLowerCase(),
      })),
    };
  },

  transformOWMForecastData(payload) {
    const city = payload?.city || {};
    const list = payload?.list || [];
    const currentEntry = list[0] || {};

    // 3-hour steps from OWM forecast, roughly next 24 hours.
    const hourly = list.slice(0, 8).map((h) => ({
      time: new Date((h.dt || 0) * 1000).toTimeString().slice(0, 5),
      temp: Math.round(h.main?.temp ?? 0),
      icon: h.weather?.[0]?.icon || "01d",
      condition: h.weather?.[0]?.main || "",
      pop: Math.round((h.pop ?? 0) * 100),
      rainProbability: Math.round((h.pop ?? 0) * 100),
      aqi: null,
    }));

    const groupedByDate = list.reduce((acc, entry) => {
      const date = new Date((entry.dt || 0) * 1000).toISOString().split("T")[0];
      if (!acc[date]) {
        acc[date] = [];
      }
      acc[date].push(entry);
      return acc;
    }, {});

    const daily = Object.entries(groupedByDate)
      .slice(0, 7)
      .map(([date, entries]) => {
        const temps = entries.map((e) => e.main?.temp).filter((v) => typeof v === "number");
        const iconSource = entries.find((e) => String(e.dt_txt || "").includes("12:00:00")) || entries[0];

        return {
          date,
          tempMax: Math.round(Math.max(...temps)),
          tempMin: Math.round(Math.min(...temps)),
          icon: iconSource?.weather?.[0]?.icon || "01d",
          condition: iconSource?.weather?.[0]?.main || "",
        };
      });

    return {
      city: city?.name || "Unknown",
      current: {
        temp: Math.round(currentEntry?.main?.temp ?? 0),
        feelsLike: Math.round(currentEntry?.main?.feels_like ?? 0),
        humidity: currentEntry?.main?.humidity ?? 0,
        wind: Math.round((currentEntry?.wind?.speed ?? 0) * 3.6 * 10) / 10,
        visibility: Number(((currentEntry?.visibility ?? 10000) / 1000)).toFixed(1),
        pressure: currentEntry?.main?.pressure ?? 0,
        condition: currentEntry?.weather?.[0]?.main || "Clear",
        description: currentEntry?.weather?.[0]?.description || "clear sky",
        icon: currentEntry?.weather?.[0]?.icon || "01d",
        uvi: 0,
        aqi: null,
        sunrise: city?.sunrise || 0,
        sunset: city?.sunset || 0,
      },
      hourly,
      daily,
      alerts: [],
    };
  },
};
