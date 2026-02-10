import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { MapPin, Navigation, Sparkles, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
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

export default function Home() {
  const { location, currentLocation, savedLocation, loading: locationLoading, error: locationError, refreshLocation } = useLocation();
  const { data, loading, error } = useWeather(location?.lat, location?.lon);
  const { isLoggedIn, user } = useAuth();
  const [insight, setInsight] = useState(null);
  const [insightLoading, setInsightLoading] = useState(false);
  const [showLocationSelector, setShowLocationSelector] = useState(false);

  // Fetch AI quick insight
  useEffect(() => {
    if (data) {
      setInsightLoading(true);
      const insightData = {
        persona: user?.persona || 'general',
        weatherRisks: user?.weatherRisks || [],
        weatherData: data,
        location: location
      };
      
      insightService
        .getQuickInsight(insightData)
        .then((res) => {
          setInsight(res);
          setInsightLoading(false);
        })
        .catch(() => {
          setInsightLoading(false);
        });
    }
  }, [data, user, location]);

  // Handle location selection
  const handleLocationSelect = (newLocation) => {
    // Refresh location data
    refreshLocation();
  };

  if (loading || locationLoading) return <Loader />;

  if (error && !data) {
    return (
      <div className="py-20 text-center">
        <p className="text-lg text-red-400">⚠️ {error}</p>
        <p className="mt-2 text-sm text-gray-500">Please check your internet connection</p>
        <button 
          onClick={refreshLocation}
          className="px-4 py-2 mt-4 text-white transition-colors bg-blue-500 rounded-lg hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
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
      className="space-y-4 sm:space-y-6"
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
          <div className="flex items-center gap-2 mt-3">
            <div className="flex items-center gap-2">
              {savedLocation ? (
                <MapPin className="w-4 h-4 text-blue-400" />
              ) : (
                <Navigation className="w-4 h-4 text-green-400" />
              )}
              <span className="text-sm text-gray-300">
                {location.city}
                {savedLocation && isLoggedIn && (
                  <span className="ml-1 text-blue-400">(Saved)</span>
                )}
                {!savedLocation && (
                  <span className="ml-1 text-green-400">(Current)</span>
                )}
              </span>
            </div>
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
          <GlassCard className="p-4 border-indigo-500/20 bg-gradient-to-r from-indigo-500/[0.08] to-purple-500/[0.08] m-5">
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
                <div className="flex items-center justify-center flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="mb-1 text-xs font-semibold tracking-wider text-indigo-300 uppercase">
                    {insight.title || 'AI Insight'}
                  </p>
                  <p className="text-sm leading-relaxed text-gray-200">
                    {insight.message || insight}
                  </p>
                  {isLoggedIn && (
                    <Link 
                      to="/planner"
                      className="inline-flex items-center gap-1 mt-2 text-xs font-medium text-indigo-400 transition-colors hover:text-indigo-300"
                    >
                      Plan your day with AI
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
        <WeatherCard data={data?.current} city={data?.city} />
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
