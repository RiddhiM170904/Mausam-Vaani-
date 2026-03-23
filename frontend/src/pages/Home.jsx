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

const looksGenericInsight = (text) => /routine|balanced|stable|normal|manageable|check weather|stay updated/i.test(String(text || ""));

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

const buildCuratedHomeInsight = (weatherData, name) => {
  const temp = Number(weatherData?.current?.temp || 0);
  const aqi = Number(weatherData?.current?.aqi || 0);
  const rain = normalizeRainRisk(weatherData);
  const hour = new Date().getHours();
  const slot = hour < 12 ? "morning" : hour < 17 ? "afternoon" : "evening";
  const safeName = name || "friend";

  if (rain >= 0.55) {
    return `Hey ${safeName} 👋 ${slot === "evening" ? "shaam ko weather thoda change ho sakta hai" : "aaj baarish ka chance zyada hai"} 🌧️ Umbrella/raincoat saath rakho aur commute me thoda buffer rakhna 👍`;
  }
  if (temp >= 34 && aqi >= 130) {
    return `Hey ${safeName} 👋 aaj heat ke saath hawa bhi heavy feel ho sakti hai 🌫️ Paani saath rakho, mask use karo aur long outdoor stay avoid karo 👍`;
  }
  if (temp >= 34) {
    return `Hey ${safeName} 👋 aaj dhoop thodi strong rahegi ☀️ Bahar jao toh hydration maintain rakho aur 12-4 PM direct sun exposure avoid karo 👍`;
  }
  if (aqi >= 130) {
    return `Hey ${safeName} 👋 aaj air quality thodi heavy feel ho sakti hai 🌫️ Outdoor time limit rakho aur zarurat ho to mask carry karo 👍`;
  }
  return `Hey ${safeName} 👋 aaj weather kaafi theek lag raha hai 🙂 Normal plans continue kar sakte ho, bas paani saath rakhna 👍`;
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
  const insightInFlightRef = useRef(false);
  const lastInsightKeyRef = useRef("");
  const insightMessage = getSafeInsightMessage(insight);
  const insightLines = getInsightLines(insight);
  const insightTips = getInsightTips(insight);
  const finalInsightMessage = useMemo(() => {
    if (!isLoggedIn || !data) return insightMessage;
    if (!insightMessage || looksGenericInsight(insightMessage)) {
      return buildCuratedHomeInsight(data, user?.name || user?.full_name || "friend");
    }
    return insightMessage;
  }, [isLoggedIn, data, insightMessage, user?.name, user?.full_name]);
  const finalInsightLines = useMemo(() => {
    return String(finalInsightMessage || "")
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);
  }, [finalInsightMessage]);
  const displayLocationName =
    location?.city && location.city !== "Unknown" ? location.city : (data?.city || "Unknown");

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
      Math.round(Number(data?.current?.aqi || 0) / 25),
      String(data?.current?.condition || '').toLowerCase(),
      Math.floor(new Date().getHours() / 3),
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
  }, [isLoggedIn, data, user?.id, user?.persona, user?.weather_risks, user?.weatherRisks, location?.lat, location?.lon, location?.city]);

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
