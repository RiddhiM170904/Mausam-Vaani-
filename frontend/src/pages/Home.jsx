import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { MapPin, Sparkles, ArrowRight } from "lucide-react";
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

export default function Home() {
  const navigate = useNavigate();
  const { location, currentLocation, savedLocation, locationSource, loading: locationLoading, error: locationError, refreshLocation } = useLocation();
  const { data, loading, error } = useWeather(location?.lat, location?.lon);
  const { isLoggedIn, user } = useAuth();
  const [insight, setInsight] = useState(null);
  const [insightLoading, setInsightLoading] = useState(false);
  const [showLocationSelector, setShowLocationSelector] = useState(false);
  const insightInFlightRef = useRef(false);
  const lastInsightKeyRef = useRef("");
  const insightMessage = getSafeInsightMessage(insight);
  const displayLocationName =
    location?.city && location.city !== "Unknown" ? location.city : (data?.city || "Unknown");

  // Fetch AI quick insight
  useEffect(() => {
    if (!data) return;

    const requestKey = [
      location?.lat,
      location?.lon,
      user?.persona || "general",
      data?.current?.temp,
      data?.current?.condition,
    ].join("|");

    if (insightInFlightRef.current || lastInsightKeyRef.current === requestKey) {
      return;
    }

    insightInFlightRef.current = true;
    lastInsightKeyRef.current = requestKey;
    setInsightLoading(true);

    const insightData = {
      persona: user?.persona || "general",
      weatherRisks: user?.weather_risks || user?.weatherRisks || [],
      weatherData: data,
      location,
      latitude: location?.lat,
      longitude: location?.lon,
      location_name: location?.city,
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
  }, [data, user, location]);

  useEffect(() => {
    if (isLoggedIn && user && user.planner_profile_completed === false) {
      navigate("/insights");
    }
  }, [isLoggedIn, user, navigate]);

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

      {/* AI Quick Insight Card */}
      {(insight || insightLoading) && (
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
                  <p className="text-sm leading-relaxed text-gray-200">
                    {insightMessage}
                  </p>
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
