import { useState, useEffect, useRef, useMemo } from "react";
import { motion } from "framer-motion";
import { MapPin, Sparkles, ArrowRight, UserPlus, LogIn, X } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import { useAuth } from "../context/AuthContext";
import { insightService } from "../services/insightService";
import { getGreeting } from "../utils/helpers";

import WeatherCard from "../components/WeatherCard";
import HourlySlider from "../components/HourlySlider";
import ForecastCard from "../components/ForecastCard";
import AlertBanner from "../components/AlertBanner";
import WeatherDetails from "../components/WeatherDetails";
import LocationSelector from "../components/LocationSelector";
import Loader from "../components/Loader";
import GlassCard from "../components/GlassCard";
import ServiceUnavailable from "../components/ServiceUnavailable";

const GUEST_ONBOARDING_KEY = "mv_guest_onboarding_seen";

const normalizeRainRisk = (weatherData) => {
  const current = weatherData?.current || {};
  const fromCurrent = Number(current?.rainProbability ?? current?.rain_probability);
  const fromHourly = Number(weatherData?.hourly?.[0]?.rainProbability ?? weatherData?.hourly?.[0]?.pop);
  const raw = Number.isFinite(fromCurrent) ? fromCurrent : fromHourly;
  if (!Number.isFinite(raw)) return 0;
  return raw > 1 ? raw / 100 : raw;
};

const INSIGHT_REFRESH_INTERVAL_MS = 20 * 60 * 1000;
const LAST_HOME_INSIGHT_KEY = "mv_home_dynamic_insight_last_v1";

const formatInsightUpdatedAt = (value) => {
  const ms = Number(value);
  if (!Number.isFinite(ms) || ms <= 0) return null;

  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
};

const looksGenericInsight = (text) =>
  /routine|balanced|stable|normal|manageable|check weather|stay updated|overall manageable|continue karo/i.test(
    String(text || "")
  );

const INSIGHT_ACTION_WORDS = [
  "carry",
  "avoid",
  "mask",
  "umbrella",
  "hydration",
  "paani",
  "travel",
  "buffer",
  "cap",
  "shade",
  "break",
  "slow",
  "jaldi",
  "plan",
  "rakhna",
];

const INSIGHT_SITUATION_WORDS = [
  "rain",
  "baarish",
  "aqi",
  "air",
  "hawa",
  "heat",
  "garmi",
  "humidity",
  "wind",
  "fog",
  "mist",
  "dhoop",
  "temp",
  "temperature",
];

const INSIGHT_TIME_WORDS = [
  "abhi",
  "subah",
  "shaam",
  "raat",
  "morning",
  "afternoon",
  "evening",
  "night",
  "12-4",
  "time",
  "today",
  "aaj",
];

const USER_ACTION_BY_PERSONA = {
  driver: "commute",
  delivery: "delivery run",
  farmer: "field work",
  worker: "outdoor kaam",
  student: "travel",
  senior: "daily outing",
  office: "office commute",
  business_owner: "shop operations",
  commuter: "travel plan",
  general: "daily plan",
};

function includesAny(text, words) {
  const value = String(text || "").toLowerCase();
  return words.some((word) => value.includes(word));
}

function getInsightQualityScore(text) {
  const value = String(text || "").trim();
  if (!value) return 0;

  let score = 0;
  const words = value.split(/\s+/).filter(Boolean);

  if (words.length >= 14) score += 0.2;
  if (words.length >= 20) score += 0.1;
  if (includesAny(value, INSIGHT_SITUATION_WORDS)) score += 0.3;
  if (includesAny(value, INSIGHT_ACTION_WORDS)) score += 0.25;
  if (includesAny(value, INSIGHT_TIME_WORDS)) score += 0.15;

  if (looksGenericInsight(value)) score -= 0.3;
  if (/weather mostly stable|normal plans continue|manageable lag raha/i.test(value)) score -= 0.2;

  return Math.max(0, Math.min(1, score));
}

function shouldReplaceWithDynamicInsight(text) {
  if (!String(text || "").trim()) return true;
  if (looksGenericInsight(text)) return true;
  return getInsightQualityScore(text) < 0.55;
}

function classifyPrimaryCondition(weatherData) {
  const current = weatherData?.current || {};
  const temp = Number(current?.temp || 0);
  const feelsLike = Number(current?.feels_like ?? current?.feelsLike ?? temp);
  const aqi = Number(current?.aqi || 0);
  const humidity = Number(current?.humidity || 0);
  const wind = Number(current?.wind || 0);
  const condition = String(current?.condition || "").toLowerCase();
  const rain = normalizeRainRisk(weatherData);

  const rainy = rain >= 0.35 || /rain|storm|drizzle|thunder/.test(condition);
  const polluted = aqi >= 110 || /haze|smoke|fog|mist/.test(condition);
  const hot = Math.max(temp, feelsLike) >= 32;

  if (rainy && wind >= 22) return "storm";
  if (rainy && polluted) return "rain_pollution";
  if (hot && polluted) return "heat_pollution";
  if (rainy) return "rain";
  if (hot) return "heat";
  if (polluted) return "pollution";
  if (wind >= 22) return "wind";
  if (humidity >= 82) return "humidity";
  if (Math.min(temp, feelsLike) <= 12) return "cold";
  return "stable";
}

function getTimeContext(hour = new Date().getHours()) {
  if (hour < 5) return { slot: "late_night", phrase: "raat me" };
  if (hour < 10) return { slot: "morning", phrase: "subah" };
  if (hour < 13) return { slot: "midday", phrase: "dopahar me" };
  if (hour < 17) return { slot: "afternoon", phrase: "abhi" };
  if (hour < 21) return { slot: "evening", phrase: "shaam me" };
  return { slot: "night", phrase: "raat me" };
}

function getTempLevel(tempLike) {
  if (tempLike >= 40) return "kaafi tez garmi";
  if (tempLike >= 35) return "strong heat";
  if (tempLike >= 31) return "thodi garmi";
  if (tempLike <= 12) return "thandi conditions";
  return "mild weather";
}

function getActionItem(condition) {
  const map = {
    heat: "paani + cap",
    rain: "umbrella/raincoat",
    pollution: "mask",
    heat_pollution: "paani + mask",
    rain_pollution: "umbrella + mask",
    storm: "rain gear + safer route",
    cold: "light jacket",
    wind: "safe speed",
    humidity: "hydration",
    stable: "basic hydration",
  };
  return map[condition] || "basic hydration";
}

function chooseVariant(options, seed) {
  if (!Array.isArray(options) || !options.length) return "";
  return options[Math.abs(seed) % options.length];
}

function buildLineTemplatesByCondition(condition) {
  const templates = {
    heat: [
      "abhi heat thodi kam ho rahi hai lekin din bhar ki garmi feel hogi.",
      "abhi dhoop kaafi strong hai.",
      "garmi zyada hai aur thakan jaldi ho sakti hai.",
    ],
    rain: [
      "shaam tak baarish aa sakti hai.",
      "rain chance high lag raha hai.",
      "road slippery ho sakti hai.",
    ],
    pollution: [
      "aaj hawa thodi heavy lag rahi hai.",
      "AQI high side pe hai.",
      "air quality perfect nahi hai.",
    ],
    heat_pollution: [
      "garmi ke saath hawa bhi heavy lag rahi hai.",
      "heat aur AQI dono elevated hain.",
      "body load zyada ho sakta hai.",
    ],
    rain_pollution: [
      "baarish ke saath hawa bhi heavy ho sakti hai.",
      "mixed weather conditions aa sakti hain.",
      "weather quick change ho sakta hai.",
    ],
    storm: [
      "rain aur hawa dono strong ho sakte hain.",
      "weather jaldi badal sakta hai.",
      "open area me extra care rakho.",
    ],
    cold: [
      "subah thodi thand aur halki fog ho sakti hai.",
      "cold weather me body slow feel kar sakti hai.",
      "raat me thand zyada lag sakti hai.",
    ],
    wind: [
      "Aaj hawa tez ho sakti hai. {user_action} me pace slow rakho.",
      "Open road par gust aa sakte hain. Speed control me rakho.",
      "Windy weather me dhyan se travel karo.",
    ],
    humidity: [
      "Aaj humidity zyada hai. Paani pe dhyan rakho.",
      "Chipchipi garmi me thakan jaldi ho sakti hai. Break lete raho.",
      "Weather sticky lag sakta hai. Light kapde pehno.",
    ],
    stable: [
      "abhi travel ke liye time better hai.",
      "subah weather kaafi manageable hai.",
      "aaj weather theek hai aur plans smooth rahenge.",
    ],
  };

  return templates[condition] || templates.stable;
}

function buildPersonaTail(persona) {
  const tone = {
    driver: [
      "Drive smooth rakho aur sudden brake avoid karo.",
      "Road dekhkar safe speed me chalo.",
    ],
    delivery: [
      "Route simple rakho taki delay kam ho.",
      "Delivery ke beech paani peete raho.",
    ],
    student: [
      "Travel se pehle essentials ready rakho.",
      "Travel ke liye thoda jaldi niklo.",
    ],
    worker: [
      "Outdoor kaam me beech-beech me break lo.",
      "Kaam ka pace halka rakho.",
    ],
    senior: [
      "Aaj comfort ko priority do.",
      "Bahar kam time ke liye niklo.",
    ],
    farmer: [
      "Field ka kaam mausam dekhkar karo.",
      "Khuli jagah ka kaam chhote slots me rakho.",
    ],
    business_owner: [
      "Aaj customer flow weather se change ho sakta hai.",
      "Shop prep pehle se ready rakho.",
    ],
  };

  return tone[persona] || [
    "Aaj ka plan simple rakho.",
    "Weather change ho to plan adjust kar lena.",
  ];
}

function buildTimeTail(timeSlot) {
  const timeTails = {
    morning: [
      "Subah ka time kaam ke liye achha hai.",
      "Important kaam subah kar lo.",
    ],
    midday: [
      "Dopahar me bahar kam time raho.",
      "Is time me break lete raho.",
    ],
    afternoon: [
      "Afternoon me pace halka rakho.",
      "Travel time pe dhyan rakho.",
    ],
    evening: [
      "Shaam me weather better ho sakta hai.",
      "Nikalne se pehle ek quick check kar lo.",
    ],
    night: [
      "Raat me safe travel pe dhyan rakho.",
      "Light activity best rahegi.",
    ],
    late_night: [
      "Late night me unnecessary travel avoid karo.",
      "Safe rehkar hi niklo.",
    ],
  };

  return timeTails[timeSlot] || timeTails.afternoon;
}

function simplifyFallbackLanguage(text) {
  return String(text || "")
    .replace(/\bwindow\b/gi, "time")
    .replace(/\bbuffer\b/gi, "extra time")
    .replace(/major risk trigger/gi, "weather issue")
    .replace(/rush\s*window/gi, "bheed wala time")
    .replace(/comparetively/gi, "comparatively")
    .replace(/\s+/g, " ")
    .trim();
}

function avoidImmediateRepeat(nextText) {
  const message = String(nextText || "").trim();
  if (!message) return message;

  try {
    const previous = localStorage.getItem(LAST_HOME_INSIGHT_KEY) || "";
    if (previous && previous === message) return message;
    localStorage.setItem(LAST_HOME_INSIGHT_KEY, message);
  } catch {
    // ignore localStorage failures for repeat guard.
  }

  return message;
}

const buildProfileRequirements = (user) => {
  const persona = String(user?.persona || "general");
  const risks = Array.isArray(user?.weather_risks) ? user.weather_risks.join(", ") : "none";
  return [
    `User persona: ${persona}.`,
    `Risk preferences: ${risks}.`,
    "Give 2 short Hinglish lines in WhatsApp tone with clear actions.",
    "Avoid generic statements.",
  ].join(" ");
};

const buildCuratedHomeInsight = (weatherData, user = {}) => {
  const name = user?.name || user?.full_name || "friend";
  const persona = String(user?.persona || "general").toLowerCase();
  const temp = Number(weatherData?.current?.temp || 0);
  const feelsLike = Number(weatherData?.current?.feels_like ?? weatherData?.current?.feelsLike ?? temp);
  const aqi = Number(weatherData?.current?.aqi || 0);
  const humidity = Number(weatherData?.current?.humidity || 0);
  const wind = Number(weatherData?.current?.wind || 0);
  const rain = normalizeRainRisk(weatherData);

  const thermalStress = Math.max(temp, feelsLike);
  const primaryCondition = classifyPrimaryCondition(weatherData);
  const timeContext = getTimeContext();
  const tempLevel = getTempLevel(thermalStress);
  const userAction = USER_ACTION_BY_PERSONA[persona] || USER_ACTION_BY_PERSONA.general;
  const actionItem = getActionItem(primaryCondition);

  const lineTemplates = buildLineTemplatesByCondition(primaryCondition);
  const personaTails = buildPersonaTail(persona);
  const timeTails = buildTimeTail(timeContext.slot);

  const seedBase =
    Math.round(thermalStress) +
    Math.round(rain * 100) +
    Math.round(aqi / 5) +
    Math.round(wind) +
    Math.round(humidity / 2) +
    new Date().getHours() +
    Math.floor(new Date().getMinutes() / 20);

  const opener = `Hey ${name} 👋`;

  const line1Template = chooseVariant(lineTemplates, seedBase + 3);
  const line2Template = chooseVariant(personaTails, seedBase + 7);
  const line3Template = chooseVariant(timeTails, seedBase + 11);

  const line1 = String(line1Template)
    .replace(/\{time_phrase\}/g, timeContext.phrase)
    .replace(/\{temp_level\}/g, tempLevel)
    .replace(/\{action_item\}/g, actionItem)
    .replace(/\{user_action\}/g, userAction)
    .replace(/\s+/g, " ")
    .trim();

  const actionLine =
    primaryCondition === "rain" || primaryCondition === "rain_pollution" || primaryCondition === "storm"
      ? "Bahar nikalte time umbrella le jaana aur travel me thoda extra time rakhna."
      : primaryCondition === "heat" || primaryCondition === "heat_pollution"
        ? "Agar bahar ja rahe ho toh paani saath rakho aur direct dhoop avoid karo."
        : primaryCondition === "pollution"
          ? "Bahar jao toh mask use karo aur hydration maintain rakhna."
          : primaryCondition === "cold"
            ? "Bahar jao toh warm kapde pehenna aur thoda extra time rakhna."
            : "Bas paani saath rakho aur zarurat ho toh weather ek baar check kar lena.";

  const dynamicMessage = `${opener} ${line1} ${actionLine} ${line2Template}`
    .replace(/\s+/g, " ")
    .trim();

  return avoidImmediateRepeat(simplifyFallbackLanguage(dynamicMessage));
};

const getSafeInsightMessage = (insight) => {
  if (!insight) return "";
  if (typeof insight === "string") return insight;

  if (typeof insight?.message === "string" && insight.message.trim()) {
    return insight.message;
  }

  if (typeof insight?.recommendation === "string" && insight.recommendation.trim()) {
    return insight.recommendation;
  }

  if (Array.isArray(insight?.tips) && insight.tips.length > 0) {
    return String(insight.tips[0]);
  }

  return "Weather insight is being updated. Please check again shortly.";
};

const getInsightLines = (insight) => {
  const message = getSafeInsightMessage(insight);
  return String(message)
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
};

const getInsightTips = (insight) => {
  if (!insight || typeof insight === "string") return [];

  if (Array.isArray(insight?.tips) && insight.tips.length > 0) {
    return insight.tips.map((tip) => String(tip)).filter(Boolean).slice(0, 3);
  }

  return [];
};

export default function Home() {
  const navigate = useNavigate();
  const { location, currentLocation, savedLocation, locationSource, loading: locationLoading, error: locationError, refreshLocation } = useLocation();
  const { data, loading, error } = useWeather(location?.lat, location?.lon);
  const { isLoggedIn, user } = useAuth();
  const [insight, setInsight] = useState(null);
  const [insightLoading, setInsightLoading] = useState(false);
  const [showLocationSelector, setShowLocationSelector] = useState(false);
  const [showGuestOnboarding, setShowGuestOnboarding] = useState(false);
  const [insightTick, setInsightTick] = useState(0);
  const insightInFlightRef = useRef(false);
  const lastInsightKeyRef = useRef("");
  const insightMessage = getSafeInsightMessage(insight);
  const insightSource = String(insight?.source || "").toLowerCase();
  const insightLines = getInsightLines(insight);
  const insightTips = getInsightTips(insight);
  const finalInsightMessage = useMemo(() => {
    if (!isLoggedIn || !data) return insightMessage;
    const shouldForceTemplate = insightSource.startsWith("fallback");
    if (shouldForceTemplate || shouldReplaceWithDynamicInsight(insightMessage)) {
      return buildCuratedHomeInsight(data, user || {});
    }
    return insightMessage;
  }, [isLoggedIn, data, insightMessage, insightSource, user]);
  const finalInsightLines = useMemo(() => {
    return String(finalInsightMessage || "")
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);
  }, [finalInsightMessage]);
  const displayLocationName =
    location?.city && location.city !== "Unknown" ? location.city : (data?.city || "Unknown");
  const insightUpdatedAtLabel = useMemo(() => {
    const fromInsight = formatInsightUpdatedAt(insight?.generatedAt);
    if (fromInsight) return fromInsight;
    if (data?.lastUpdated) {
      const parsed = Date.parse(data.lastUpdated);
      return formatInsightUpdatedAt(parsed);
    }
    return null;
  }, [insight?.generatedAt, data?.lastUpdated]);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setInsightTick((prev) => prev + 1);
    }, INSIGHT_REFRESH_INTERVAL_MS);

    return () => clearInterval(intervalId);
  }, []);

  // Fetch AI quick insight
  useEffect(() => {
    if (!data || !isLoggedIn) {
      setInsight(null);
      setInsightLoading(false);
      return;
    }

    const requestKey = [
      location?.lat,
      location?.lon,
      user?.id || "guest",
      user?.persona || "general",
      Math.round(Number(data?.current?.temp || 0) / 2),
      Math.round(Number((data?.current?.rain_probability ?? data?.rain_probability) ?? 0) * 10),
      Math.round(Number(data?.current?.humidity || 0) / 5),
      Math.round(Number(data?.current?.wind || 0) / 2),
      Math.round(Number(data?.current?.aqi || 0) / 25),
      String(data?.current?.condition || '').toLowerCase(),
      `${new Date().getHours()}-${Math.floor(new Date().getMinutes() / 20)}`,
      insightTick,
    ].join("|");

    if (insightInFlightRef.current || lastInsightKeyRef.current === requestKey) {
      return;
    }

    insightInFlightRef.current = true;
    lastInsightKeyRef.current = requestKey;
    setInsightLoading(true);

    const insightData = {
      userId: user?.id || null,
      persona: user?.persona || "general",
      weatherRisks: user?.weather_risks || user?.weatherRisks || [],
      weatherData: data,
      location,
      latitude: location?.lat,
      longitude: location?.lon,
      location_name: location?.city,
      requirements: buildProfileRequirements(user),
    };

    insightService
      .getQuickInsight(insightData)
      .then((res) => {
        setInsight(res);
      })
      .catch(() => {
        // Keep silent fallback behavior for unavailable AI backend.
      })
      .finally(() => {
        insightInFlightRef.current = false;
        setInsightLoading(false);
      });
  }, [isLoggedIn, data, user?.id, user?.persona, user?.weather_risks, user?.weatherRisks, location?.lat, location?.lon, location?.city, insightTick]);

  useEffect(() => {
    if (isLoggedIn && user && user.planner_profile_completed === false) {
      navigate("/insights");
    }
  }, [isLoggedIn, user, navigate]);

  // One-time guest onboarding CTA after first location resolution.
  useEffect(() => {
    if (locationLoading || isLoggedIn || !location) {
      return;
    }

    const hasSeenOnboarding = localStorage.getItem(GUEST_ONBOARDING_KEY) === "1";
    if (!hasSeenOnboarding) {
      setShowGuestOnboarding(true);
    }
  }, [locationLoading, isLoggedIn, location]);

  const dismissGuestOnboarding = () => {
    localStorage.setItem(GUEST_ONBOARDING_KEY, "1");
    setShowGuestOnboarding(false);
  };

  // Handle location selection
  const handleLocationSelect = (newLocation) => {
    // Refresh location data
    refreshLocation();
  };

  if (loading || locationLoading) return <Loader />;

  if (error && !data) {
    return (
      <ServiceUnavailable
        title="Weather Service Unavailable"
        message="We are not servicing at this time. Sorry for the inconvenience."
        showRetry
        onRetry={refreshLocation}
      />
    );
  }

  const stagger = {
    hidden: {},
    show: { transition: { staggerChildren: 0.08 } },
  };

  const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  };

  return (
    <motion.div
      variants={stagger}
      initial="hidden"
      animate="show"
      className="p-3 space-y-4 border rounded-3xl border-indigo-400/10 bg-slate-950/30 sm:space-y-6 sm:p-4"
    >
      {/* Greeting & Location */}
      <motion.div variants={fadeUp}>
        <h2 className="text-xl font-semibold text-white sm:text-2xl">
          {getGreeting()} 👋
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          {isLoggedIn
            ? `Welcome back, ${user?.name || "User"}`
            : "Here's your live weather update"}
        </p>

        {/* Current Location Display */}
        {location && (
          <div className="mt-3">
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-gray-300">
                {location.city}
                {locationSource === "saved" && savedLocation && isLoggedIn && (
                  <span className="ml-1 text-blue-400">(Saved)</span>
                )}
              </span>
            </div>
            {location.formattedAddress && (
              <p className="mt-1 text-xs text-gray-500">{location.formattedAddress}</p>
            )}
            <button
              onClick={() => setShowLocationSelector(true)}
              className="text-xs text-blue-400 underline transition-colors hover:text-blue-300"
            >
              Change
            </button>
          </div>
        )}

        {/* Location Error */}
        {locationError && (
          <div className="p-2 mt-2 border rounded-lg bg-yellow-500/10 border-yellow-500/30">
            <p className="text-xs text-yellow-400">{locationError}</p>
          </div>
        )}
      </motion.div>

      {/* Alerts */}
      {data?.alerts?.length > 0 && (
        <motion.div variants={fadeUp}>
          <AlertBanner alerts={data.alerts} />
        </motion.div>
      )}

      {/* First-time guest onboarding popup */}
      {!isLoggedIn && showGuestOnboarding && (
        <motion.div
          variants={fadeUp}
          className="relative"
        >
          <GlassCard className="border-cyan-500/25 bg-linear-to-br from-cyan-500/10 to-indigo-500/10 p-4 sm:p-5">
            <button
              onClick={dismissGuestOnboarding}
              className="absolute right-3 top-3 rounded-lg p-1 text-gray-400 transition-colors hover:bg-white/10 hover:text-white"
              aria-label="Dismiss guest onboarding"
            >
              <X className="h-4 w-4" />
            </button>

            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-300">Welcome to Mausam Vaani</p>
            <h3 className="mt-1 text-base font-semibold text-white sm:text-lg">
              Get AI curated weather insights for your exact location
            </h3>
            <p className="mt-1 text-sm text-gray-300">
              Sign up to unlock personalized alerts, planner recommendations, and risk-aware daily guidance.
            </p>

            <div className="mt-4 flex flex-wrap items-center gap-2">
              <button
                onClick={() => {
                  dismissGuestOnboarding();
                  navigate("/signup");
                }}
                className="inline-flex items-center gap-2 rounded-xl border border-cyan-400/40 bg-cyan-500/20 px-3 py-2 text-sm font-medium text-cyan-200 transition-colors hover:bg-cyan-500/30"
              >
                <UserPlus className="h-4 w-4" />
                Sign up
              </button>
              <button
                onClick={() => {
                  dismissGuestOnboarding();
                  navigate("/login");
                }}
                className="inline-flex items-center gap-2 rounded-xl border border-indigo-400/35 bg-indigo-500/15 px-3 py-2 text-sm font-medium text-indigo-200 transition-colors hover:bg-indigo-500/25"
              >
                <LogIn className="h-4 w-4" />
                Sign in
              </button>
              <button
                onClick={dismissGuestOnboarding}
                className="text-xs text-gray-400 underline underline-offset-2 hover:text-gray-200"
              >
                Continue as guest
              </button>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* AI Quick Insight Card */}
      {isLoggedIn && (insight || insightLoading) && (
        <motion.div variants={fadeUp}>
          <GlassCard className="p-4 border-indigo-500/20 bg-linear-to-r from-indigo-500/8 to-purple-500/8">
            {insightLoading ? (
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-indigo-500/20">
                  <Sparkles className="w-5 h-5 text-indigo-400 animate-pulse" />
                </div>
                <div className="flex-1">
                  <div className="w-24 h-3 mb-2 rounded bg-gray-700/50 animate-pulse"></div>
                  <div className="w-3/4 h-4 rounded bg-gray-700/50 animate-pulse"></div>
                </div>
              </div>
            ) : insight && (
              <div className="flex items-start gap-3">
                <div className="flex items-center justify-center w-10 h-10 shrink-0 rounded-xl bg-linear-to-br from-indigo-500 to-purple-600">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="mb-1 text-xs font-semibold tracking-wider text-indigo-300 uppercase">
                    {insight.title || 'AI Insight'}
                  </p>
                  {insightUpdatedAtLabel && (
                    <p className="mb-1 text-[11px] text-gray-400">Updated at {insightUpdatedAtLabel}</p>
                  )}
                  
                  <div className="space-y-1.5">
                    {(finalInsightLines.length ? finalInsightLines : [finalInsightMessage]).map((line, idx) => (
                      <p key={idx} className="text-sm leading-relaxed text-gray-200">
                        {line}
                      </p>
                    ))}
                  </div>
                  {insightTips.length > 0 && (
                    <div className="mt-3 space-y-1">
                      {insightTips.map((tip, idx) => (
                        <p key={idx} className="text-xs leading-relaxed text-indigo-300/90">• {tip}</p>
                      ))}
                    </div>
                  )}
                  {isLoggedIn && (
                    <Link
                      to="/insights"
                      className="inline-flex items-center gap-1 mt-2 text-xs font-medium text-indigo-400 transition-colors hover:text-indigo-300"
                    >
                      Explore AI insights
                      <ArrowRight className="w-3 h-3" />
                    </Link>
                  )}
                </div>
              </div>
            )}
          </GlassCard>
        </motion.div>
      )}



      {/* Main weather card */}
      <motion.div variants={fadeUp}>
        <WeatherCard data={data?.current} city={displayLocationName} />
      </motion.div>

      {/* Hourly */}
      <motion.div variants={fadeUp}>
        <HourlySlider hours={data?.hourly} />
      </motion.div>

      {/* Details grid */}
      <motion.div variants={fadeUp}>
        <WeatherDetails data={data?.current} />
      </motion.div>

      {/* 7-day */}
      <motion.div variants={fadeUp}>
        <ForecastCard days={data?.daily} />
      </motion.div>

      {/* Location Selector Modal */}
      <LocationSelector
        isOpen={showLocationSelector}
        onClose={() => setShowLocationSelector(false)}
        onLocationSelect={handleLocationSelect}
      />
    </motion.div>
  );
}
