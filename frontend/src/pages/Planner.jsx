import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "../context/AuthContext";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import { insightService } from "../services/insightService";
import { isSupabaseConfigured, supabase } from "../services/supabaseClient";
import profileConfig from "../config/userProfileConfig.json";
import GlassCard from "../components/GlassCard";
import Loader from "../components/Loader";
import { 
  Sparkles, 
  Calendar, 
  Clock, 
  MapPin, 
  User, 
  AlertTriangle,
  CheckCircle,
  ArrowRight,
  RefreshCw,
  ChevronDown,
  Sun,
  CloudRain,
  Wind,
  Thermometer,
  Eye,
  Car,
  Briefcase,
  Plane,
  Shovel,
  PartyPopper,
  Truck,
  Dumbbell,
  MoreHorizontal
} from "lucide-react";

// Activity options with icons
const ACTIVITIES = [
  { id: "travel", label: "Travel", icon: Plane, emoji: "✈️" },
  { id: "outdoor", label: "Outdoor Work", icon: Shovel, emoji: "🏗️" },
  { id: "farming", label: "Farming", icon: Shovel, emoji: "🌾" },
  { id: "event", label: "Outdoor Event", icon: PartyPopper, emoji: "🎉" },
  { id: "delivery", label: "Delivery", icon: Truck, emoji: "📦" },
  { id: "exercise", label: "Exercise", icon: Dumbbell, emoji: "🏃" },
  { id: "commute", label: "Commute", icon: Car, emoji: "🚗" },
  { id: "other", label: "Other", icon: MoreHorizontal, emoji: "📝" },
];

// Risk priorities
const RISK_OPTIONS = [
  { id: "avoid_rain", label: "Avoid Rain", icon: CloudRain },
  { id: "avoid_heat", label: "Avoid Heat", icon: Thermometer },
  { id: "avoid_traffic", label: "Avoid Traffic", icon: Car },
  { id: "avoid_wind", label: "Avoid Wind", icon: Wind },
  { id: "avoid_fog", label: "Avoid Fog", icon: Eye },
];

// Duration options
const DURATION_OPTIONS = [
  { value: 1, label: "1 hour" },
  { value: 2, label: "2 hours" },
  { value: 4, label: "Half day" },
  { value: 8, label: "Full day" },
  { value: 10, label: "Extended" },
];

const TIME_PRESETS = [
  { id: "now", label: "Now" },
  { id: "next_2_hours", label: "Next 2 hrs" },
  { id: "evening", label: "Evening" },
  { id: "custom", label: "Custom" },
];

function toHHMM(date) {
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

const buildProfileQuestions = (persona) => {
  const personaConfig = profileConfig.user_types?.[persona] || profileConfig.user_types?.general;
  const compulsory = personaConfig?.compulsory || [];
  const optional = personaConfig?.optional || [];
  return [...compulsory, ...optional];
};

const isAnswerProvided = (value) => {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  return true;
};

const isQuestionActive = (question, answers) => {
  if (!question.depends_on) return true;
  return answers?.[question.depends_on.key] === question.depends_on.value;
};

const normalizeWeatherUnits = (value) => {
  const text = String(value || "").trim();
  if (!text) return "";

  return text
    .replace(/(-?\d+(?:\.\d+)?)\s*°?\s*C\b/gi, "$1°C")
    .replace(/\bNA\b/gi, "N/A")
    .replace(/\bkm\/?h\b/gi, "km/h");
};

const formatRecommendationSections = (recommendation) => {
  const lines = String(recommendation || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  return lines
    .map((line) => {
      const match = line.match(/^([^:]+):\s*(.+)$/);
      if (!match) return null;
      return {
        label: match[1].trim(),
        value: normalizeWeatherUnits(match[2].trim()),
      };
    })
    .filter(Boolean);
};

export default function Planner() {
  const { isLoggedIn, user, refreshProfile } = useAuth();
  const { location } = useLocation();
  const { data: weatherData, loading: weatherLoading } = useWeather(location?.lat, location?.lon);
  const [profileUserType, setProfileUserType] = useState(user?.persona || "general");
  const [profileUserTypeLabel, setProfileUserTypeLabel] = useState(user?.other_persona_text || "");

  // Form state
  const [activity, setActivity] = useState("");
  const [customActivity, setCustomActivity] = useState("");
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [timeRange, setTimeRange] = useState({ start: "09:00", end: "18:00" });
  const [timePreset, setTimePreset] = useState("custom");
  const [risks, setRisks] = useState([]);
  const [duration, setDuration] = useState(4);
  const [notes, setNotes] = useState("");

  // Planner profile state
  const [showProfileSetup, setShowProfileSetup] = useState(!user?.planner_profile_completed);
  const [profileAnswers, setProfileAnswers] = useState(user?.planner_profile?.answers || {});
  const [profileCompleted, setProfileCompleted] = useState(Boolean(user?.planner_profile_completed));
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState("");
  const [profileSuccess, setProfileSuccess] = useState("");
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showForm, setShowForm] = useState(true);

  useEffect(() => {
    if (!user) return;

    const nextPersona = user?.persona || "general";
    setProfileUserType(nextPersona);
    setProfileUserTypeLabel(user?.other_persona_text || "");
    setProfileAnswers(user?.planner_profile?.answers || {});
    setProfileCompleted(Boolean(user?.planner_profile_completed));
    setShowProfileSetup(!user?.planner_profile_completed);
    setProfileSuccess("");
  }, [user?.id, user?.persona, user?.planner_profile_completed]);

  useEffect(() => {
    if (!profileSuccess) return;

    const timeoutId = setTimeout(() => {
      setProfileSuccess("");
    }, 4000);

    return () => clearTimeout(timeoutId);
  }, [profileSuccess]);

  useEffect(() => {
    if (!isLoggedIn || !user?.id || !isSupabaseConfigured || !supabase) return;

    let isMounted = true;
    const fetchPlannerProfile = async () => {
      setProfileLoading(true);
      setProfileError("");
      setProfileSuccess("");

      try {
        const { data, error } = await supabase
          .from("users")
          .select("persona, other_persona_text, planner_profile, planner_profile_completed")
          .eq("id", user.id)
          .maybeSingle();

        if (error) {
          if (isMounted) {
            setProfileError(error.message || "Failed to load planner profile.");
          }
          return;
        }

        if (!isMounted || !data) return;

        setProfileUserType(data.persona || "general");
        setProfileUserTypeLabel(data.other_persona_text || "");
        setProfileAnswers(data?.planner_profile?.answers || {});
        setProfileCompleted(Boolean(data?.planner_profile_completed));
        setShowProfileSetup(!data?.planner_profile_completed);
      } catch {
        if (isMounted) {
          setProfileError("Failed to load planner profile. Please try again.");
        }
      } finally {
        if (isMounted) {
          setProfileLoading(false);
        }
      }
    };

    fetchPlannerProfile();

    return () => {
      isMounted = false;
    };
  }, [isLoggedIn, user?.id]);

  const profileQuestions = buildProfileQuestions(profileUserType);
  const visibleQuestions = profileQuestions.filter((question) =>
    isQuestionActive(question, profileAnswers)
  );
  const missingRequired = visibleQuestions.filter(
    (question) => question.required && !isAnswerProvided(profileAnswers[question.key])
  );
  const plannerReady = profileCompleted && !profileLoading;
  const recommendationSections = formatRecommendationSections(result?.recommendation);

  // Get persona display name
  const getPersonaLabel = (persona) => {
    const labels = {
      'general': 'General User',
      'driver': 'Driver',
      'worker': 'Outdoor Worker',
      'office_employee': 'Office Employee',
      'farmer': 'Farmer',
      'delivery': 'Delivery Personnel',
      'senior_citizen': 'Senior Citizen',
      'student': 'Student',
      'business_owner': 'Business Owner',
      'other': 'Other'
    };
    return labels[persona] || 'General User';
  };

  // Toggle risk selection
  const toggleRisk = (riskId) => {
    setRisks(prev => 
      prev.includes(riskId) 
        ? prev.filter(r => r !== riskId)
        : [...prev, riskId]
    );
  };

  const applyTimePreset = (preset) => {
    const now = new Date();
    let start = new Date(now);
    let end = new Date(now);

    if (preset === "now") {
      end = new Date(now.getTime() + 2 * 60 * 60 * 1000);
    } else if (preset === "next_2_hours") {
      start = new Date(now.getTime() + 2 * 60 * 60 * 1000);
      end = new Date(now.getTime() + 4 * 60 * 60 * 1000);
    } else if (preset === "evening") {
      start = new Date(now);
      start.setHours(18, 0, 0, 0);
      end = new Date(now);
      end.setHours(20, 0, 0, 0);
    }

    if (preset !== "custom") {
      setTimeRange({ start: toHHMM(start), end: toHHMM(end) });
    }
    setTimePreset(preset);
  };

  // Generate smart plan
  const generatePlan = async () => {
    if (!activity) return;
    
    setLoading(true);
    setResult(null);
    
    try {
      const plannerData = {
        userId: user?.id || null,
        persona: profileUserType || 'general',
        location: location,
        location_name: location?.city || weatherData?.city || 'Unknown',
        weatherData: weatherData,
        plannerProfile: {
          persona: profileUserType || 'general',
          answers: profileAnswers,
          version: profileConfig.version,
        },
        activity: activity === 'other' ? customActivity : activity,
        date: selectedDate,
        timePreset,
        timeRange: timeRange,
        risks: risks,
        duration: duration,
        notes: notes
      };

      const response = await insightService.getSmartPlan(plannerData);
      setResult(response);
      setShowForm(false);
    } catch (error) {
      console.error('Planner error:', error);
      setResult({
        success: false,
        recommendation: "Unable to generate plan. Please try again.",
        tips: []
      });
    }
    
    setLoading(false);
  };

  // Reset form
  const resetPlan = () => {
    setResult(null);
    setShowForm(true);
  };

  const updateProfileAnswer = (questionId, value) => {
    setProfileAnswers((prev) => ({
      ...prev,
      [questionId]: value,
    }));
  };

  const handleProfileSave = async () => {
    setProfileError("");
    setProfileSuccess("");

    if (!user?.id) {
      setProfileError("Please sign in to save your profile.");
      return;
    }

    if (missingRequired.length > 0) {
      setProfileError("Please answer the required questions.");
      return;
    }

    if (!isSupabaseConfigured || !supabase) {
      setProfileError("Supabase is not configured.");
      return;
    }

    setProfileSaving(true);

    try {
      const plannerProfile = {
        persona: profileUserType || "general",
        persona_label: profileUserTypeLabel || null,
        answers: profileAnswers,
        question_set_version: profileConfig.version,
        completed_at: new Date().toISOString(),
        version: profileConfig.version,
      };

      const { error } = await supabase
        .from("users")
        .update({
          planner_profile: plannerProfile,
          planner_profile_completed: true,
          planner_profile_updated_at: new Date().toISOString(),
        })
        .eq("id", user.id);

      if (error) {
        setProfileError(error.message || "Failed to save profile.");
        return;
      }

      await refreshProfile();
      setProfileCompleted(true);
      setShowProfileSetup(false);
      setProfileSuccess("Profile saved successfully.");
    } catch (saveError) {
      console.error("Profile save failed:", saveError);
      setProfileError("Failed to save profile. Please try again.");
    } finally {
      setProfileSaving(false);
    }
  };

  // If not logged in, show auth prompt
  if (!isLoggedIn) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex flex-col items-center justify-center py-20 text-center"
      >
        <div className="flex items-center justify-center w-20 h-20 mb-6 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl">
          <Sparkles className="w-10 h-10 text-white" />
        </div>
        <h2 className="mb-3 text-2xl font-bold text-white">AI Weather Planner</h2>
        <p className="max-w-sm mb-6 text-sm text-gray-400">
          Get personalized weather-smart recommendations for your activities.
          Sign in to access the AI Planner.
        </p>
        <a
          href="/login"
          className="flex items-center gap-2 px-6 py-3 font-medium text-white transition-all bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl hover:from-indigo-600 hover:to-purple-700"
        >
          Sign In to Continue
          <ArrowRight className="w-4 h-4" />
        </a>
      </motion.div>
    );
  }

  if (weatherLoading) return <Loader text="Loading weather data..." />;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-2xl mx-auto space-y-6"
    >
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Sparkles className="w-6 h-6 text-indigo-400" />
          <h1 className="text-2xl font-bold text-white">AI Weather Planner</h1>
        </div>
        <p className="text-sm text-gray-400">Plan your day smarter with weather insights</p>
      </div>

      {/* User Context Card */}
      <GlassCard className="p-4">
        <h3 className="mb-3 text-xs font-semibold tracking-wider text-gray-500 uppercase">Your Profile</h3>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="flex flex-col items-center gap-1">
            <User className="w-5 h-5 text-indigo-400" />
            <span className="text-xs text-gray-400">Persona</span>
            <span className="text-sm font-medium text-white">{getPersonaLabel(profileUserType)}</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <MapPin className="w-5 h-5 text-blue-400" />
            <span className="text-xs text-gray-400">Location</span>
            <span className="text-sm text-white font-medium truncate max-w-[100px]">{location?.city || 'Unknown'}</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <Thermometer className="w-5 h-5 text-orange-400" />
            <span className="text-xs text-gray-400">Now</span>
            <span className="text-sm font-medium text-white">{weatherData?.current?.temp || '--'}°C</span>
          </div>
        </div>
      </GlassCard>

      {isLoggedIn && (
        <GlassCard className="p-5">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-gray-200">Planner profile</h3>
              <p className="text-xs text-gray-500">
                Answer a few questions so the AI can tailor predictions.
              </p>
              <p className="mt-1 text-xs text-indigo-300">
                User type: {getPersonaLabel(profileUserType)}{profileUserType === "other" && profileUserTypeLabel ? ` (${profileUserTypeLabel})` : ""}
              </p>
            </div>
            {profileCompleted && !showProfileSetup && (
              <button
                onClick={() => {
                  setProfileSuccess("");
                  setShowProfileSetup(true);
                }}
                className="px-3 py-1.5 text-xs font-semibold text-indigo-300 bg-indigo-500/10 border border-indigo-500/30 rounded-lg hover:bg-indigo-500/20"
              >
                Edit
              </button>
            )}
          </div>

          {profileLoading && (
            <div className="mt-4">
              <Loader text="Loading profile questions..." />
            </div>
          )}

          {showProfileSetup && !profileLoading && (
            <div className="mt-4 space-y-4">
              <div className="grid grid-cols-1 gap-4 px-1 sm:grid-cols-2 sm:gap-3 sm:px-0">
                {visibleQuestions.map((question) => (
                  <div key={question.key} className="space-y-1">
                    <label className="text-xs text-gray-400">
                      {question.label}
                      {question.required ? (
                        <span className="text-red-400"> *</span>
                      ) : (
                        <span className="text-gray-500"> </span>
                      )}
                    </label>

                    {question.type === "select" && (
                      <select
                        value={profileAnswers[question.key] || ""}
                        onChange={(e) => updateProfileAnswer(question.key, e.target.value)}
                        className="w-full px-3 py-2 text-sm text-white border border-gray-700 rounded-lg bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      >
                        <option value="" disabled>Select</option>
                        {question.options?.map((option) => (
                          <option key={option} value={option}>
                            {option.replace(/_/g, " ")}
                          </option>
                        ))}
                      </select>
                    )}

                    {question.type === "boolean" && (
                      <select
                        value={profileAnswers[question.key] === true ? "yes" : profileAnswers[question.key] === false ? "no" : ""}
                        onChange={(e) => updateProfileAnswer(question.key, e.target.value === "yes")}
                        className="w-full px-3 py-2 text-sm text-white border border-gray-700 rounded-lg bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      >
                        <option value="" disabled>Select</option>
                        <option value="yes">Yes</option>
                        <option value="no">No</option>
                      </select>
                    )}

                    {question.type === "text" && (
                      <input
                        type="text"
                        value={profileAnswers[question.key] || ""}
                        onChange={(e) => updateProfileAnswer(question.key, e.target.value)}
                        placeholder={question.placeholder || ""}
                        className="w-full px-3 py-2 text-sm text-white placeholder-gray-500 border border-gray-700 rounded-lg bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                    )}

                    {question.type === "number" && (
                      <div className="relative">
                        <input
                          type="number"
                          value={profileAnswers[question.key] ?? ""}
                          min={question.min}
                          max={question.max}
                          step={question.step}
                          onChange={(e) => {
                            const nextValue = e.target.value === "" ? "" : Number(e.target.value);
                            updateProfileAnswer(question.key, Number.isNaN(nextValue) ? "" : nextValue);
                          }}
                          className="w-full px-3 py-2 pr-12 text-sm text-white placeholder-gray-500 border border-gray-700 rounded-lg bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        />
                        {question.unit && (
                          <span className="absolute text-xs text-gray-500 -translate-y-1/2 right-3 top-1/2">
                            {question.unit}
                          </span>
                        )}
                      </div>
                    )}

                    {question.type === "range" && (
                      <div className="space-y-2">
                        <input
                          type="range"
                          min={question.min}
                          max={question.max}
                          step={question.step}
                          value={profileAnswers[question.key] ?? question.min}
                          onChange={(e) => updateProfileAnswer(question.key, Number(e.target.value))}
                          className="w-full accent-indigo-500"
                        />
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span>{question.min}</span>
                          <span className="font-medium text-gray-300">
                            {profileAnswers[question.key] !== undefined
                              ? `${profileAnswers[question.key]}${question.unit ? ` ${question.unit}` : ""}`
                              : "Not set"}
                          </span>
                          <span>{question.max}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {missingRequired.length > 0 && (
                <p className="text-xs text-yellow-400">
                  {missingRequired.length} required field{missingRequired.length === 1 ? "" : "s"} remaining.
                </p>
              )}

              {profileError && (
                <p className="text-xs text-red-400">{profileError}</p>
              )}

              <div className="flex flex-col gap-3 sm:flex-row">
                <button
                  onClick={handleProfileSave}
                  disabled={profileSaving || missingRequired.length > 0}
                  className="flex-1 py-2.5 rounded-lg bg-indigo-500 text-white text-sm font-semibold hover:bg-indigo-600 transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {profileSaving ? "Saving..." : "Save profile"}
                </button>
                {profileCompleted && (
                  <button
                    onClick={() => setShowProfileSetup(false)}
                    className="flex-1 py-2.5 rounded-lg bg-gray-800 text-gray-200 text-sm font-semibold border border-gray-700 hover:bg-gray-700 transition-colors"
                  >
                    Cancel
                  </button>
                )}
              </div>
            </div>
          )}

          {profileSuccess && (
            <p className="mt-4 text-xs text-green-400">{profileSuccess}</p>
          )}

          {!plannerReady && !showProfileSetup && (
            <div className="p-3 mt-4 border rounded-lg border-yellow-500/20 bg-yellow-500/10">
              <p className="text-xs text-yellow-300">
                Complete your planner profile to unlock AI planning.
              </p>
            </div>
          )}
        </GlassCard>
      )}

      <AnimatePresence mode="wait">
        {showForm && plannerReady ? (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-5"
          >
            {/* Activity Selection */}
            <GlassCard className="p-5">
              <h3 className="flex items-center gap-2 mb-4 text-sm font-semibold text-gray-300">
                <Briefcase className="w-4 h-4 text-indigo-400" />
                What are you planning?
              </h3>
              <div className="grid grid-cols-4 gap-2">
                {ACTIVITIES.map((act) => (
                  <button
                    key={act.id}
                    onClick={() => setActivity(act.id)}
                    className={`p-3 rounded-xl text-center transition-all border ${
                      activity === act.id
                        ? "bg-indigo-500/20 border-indigo-500/50 text-white"
                        : "bg-gray-800/50 border-gray-700/50 text-gray-400 hover:bg-gray-700/50 hover:text-white"
                    }`}
                  >
                    <span className="block mb-1 text-xl">{act.emoji}</span>
                    <span className="text-[10px] font-medium">{act.label}</span>
                  </button>
                ))}
              </div>
              
              {/* Custom activity input */}
              {activity === 'other' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-3"
                >
                  <input
                    type="text"
                    value={customActivity}
                    onChange={(e) => setCustomActivity(e.target.value)}
                    placeholder="Describe your activity..."
                    className="w-full px-4 py-3 text-sm text-white placeholder-gray-500 border border-gray-700 bg-gray-800/50 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </motion.div>
              )}
            </GlassCard>

            {/* Date and Time */}
            <GlassCard className="p-5">
              <h3 className="flex items-center gap-2 mb-4 text-sm font-semibold text-gray-300">
                <Calendar className="w-4 h-4 text-indigo-400" />
                When?
              </h3>
              <div className="flex flex-wrap gap-2 mb-4">
                {TIME_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => applyTimePreset(preset.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
                      timePreset === preset.id
                        ? "bg-indigo-500/20 border-indigo-500/50 text-indigo-200"
                        : "bg-gray-800/50 border-gray-700/50 text-gray-400 hover:bg-gray-700/50 hover:text-white"
                    }`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                {/* Date picker */}
                <div>
                  <label className="block mb-1 text-xs text-gray-500">Date</label>
                  <input
                    type="date"
                    value={selectedDate}
                    onChange={(e) => setSelectedDate(e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                    className="w-full px-3 py-2 text-sm text-white border border-gray-700 rounded-lg bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                
                {/* Start time */}
                <div>
                  <label className="block mb-1 text-xs text-gray-500">Start Time</label>
                  <input
                    type="time"
                    value={timeRange.start}
                    onChange={(e) => {
                      setTimePreset("custom");
                      setTimeRange(prev => ({ ...prev, start: e.target.value }));
                    }}
                    className="w-full px-3 py-2 text-sm text-white border border-gray-700 rounded-lg bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                
                {/* End time */}
                <div>
                  <label className="block mb-1 text-xs text-gray-500">End Time</label>
                  <input
                    type="time"
                    value={timeRange.end}
                    onChange={(e) => {
                      setTimePreset("custom");
                      setTimeRange(prev => ({ ...prev, end: e.target.value }));
                    }}
                    className="w-full px-3 py-2 text-sm text-white border border-gray-700 rounded-lg bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
              </div>
            </GlassCard>

            {/* Risk Priorities */}
            <GlassCard className="p-5">
              <h3 className="flex items-center gap-2 mb-4 text-sm font-semibold text-gray-300">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                What to avoid? <span className="text-xs font-normal text-gray-500">(optional)</span>
              </h3>
              <div className="flex flex-wrap gap-2">
                {RISK_OPTIONS.map((risk) => {
                  const Icon = risk.icon;
                  const isSelected = risks.includes(risk.id);
                  return (
                    <button
                      key={risk.id}
                      onClick={() => toggleRisk(risk.id)}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all border ${
                        isSelected
                          ? "bg-yellow-500/20 border-yellow-500/50 text-yellow-300"
                          : "bg-gray-800/50 border-gray-700/50 text-gray-400 hover:bg-gray-700/50 hover:text-white"
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      {risk.label}
                    </button>
                  );
                })}
              </div>
            </GlassCard>

            {/* Duration */}
            <GlassCard className="p-5">
              <h3 className="flex items-center gap-2 mb-4 text-sm font-semibold text-gray-300">
                <Clock className="w-4 h-4 text-indigo-400" />
                Duration <span className="text-xs font-normal text-gray-500">(optional)</span>
              </h3>
              <div className="flex flex-wrap gap-2">
                {DURATION_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setDuration(opt.value)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all border ${
                      duration === opt.value
                        ? "bg-indigo-500/20 border-indigo-500/50 text-white"
                        : "bg-gray-800/50 border-gray-700/50 text-gray-400 hover:bg-gray-700/50 hover:text-white"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </GlassCard>

            {/* Notes */}
            <GlassCard className="p-5">
              <h3 className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-300">
                <MoreHorizontal className="w-4 h-4 text-indigo-400" />
                Additional Notes <span className="text-xs font-normal text-gray-500">(optional)</span>
              </h3>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="E.g., Carrying equipment, elderly passengers, kids..."
                rows={2}
                className="w-full px-4 py-3 text-sm text-white placeholder-gray-500 border border-gray-700 resize-none bg-gray-800/50 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </GlassCard>

            {/* Generate Button */}
            <button
              onClick={generatePlan}
              disabled={!activity || loading || (activity === 'other' && !customActivity)}
              className="flex items-center justify-center w-full gap-2 py-4 font-semibold text-white transition-all bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl hover:from-indigo-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  Analyzing Weather...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Generate Smart Plan
                </>
              )}
            </button>
          </motion.div>
        ) : (
          /* Results View */
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-5"
          >
            {result && (
              <>
                {/* Main Recommendation */}
                <GlassCard className={`p-6 border-2 ${
                  result.riskLevel === 'High' 
                    ? 'border-red-500/30 bg-red-500/5'
                    : result.riskLevel === 'Medium'
                    ? 'border-yellow-500/30 bg-yellow-500/5'
                    : 'border-green-500/30 bg-green-500/5'
                }`}>
                  <div className="flex items-start gap-4">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${
                      result.riskLevel === 'High' 
                        ? 'bg-red-500/20'
                        : result.riskLevel === 'Medium'
                        ? 'bg-yellow-500/20'
                        : 'bg-green-500/20'
                    }`}>
                      {result.riskLevel === 'High' ? (
                        <AlertTriangle className="w-6 h-6 text-red-400" />
                      ) : result.riskLevel === 'Medium' ? (
                        <AlertTriangle className="w-6 h-6 text-yellow-400" />
                      ) : (
                        <CheckCircle className="w-6 h-6 text-green-400" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                          result.riskLevel === 'High' 
                            ? 'bg-red-500/20 text-red-300'
                            : result.riskLevel === 'Medium'
                            ? 'bg-yellow-500/20 text-yellow-300'
                            : 'bg-green-500/20 text-green-300'
                        }`}>
                          {result.riskLevel} Risk
                        </span>
                      </div>
                      {recommendationSections.length > 0 ? (
                        <div className="space-y-2">
                          {recommendationSections.map((section) => (
                            <div key={section.label} className="text-sm leading-relaxed text-gray-200">
                              <span className="font-semibold text-white">{section.label}:</span>{" "}
                              <span>{section.value}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="font-medium leading-relaxed text-white whitespace-pre-line">
                          {normalizeWeatherUnits(result.recommendation)}
                        </p>
                      )}
                    </div>
                  </div>
                </GlassCard>

                {/* Time Recommendations */}
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  {result.bestTime && (
                    <GlassCard className="p-4 border-green-500/20 bg-green-500/5">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span className="text-xs font-semibold text-green-400">Best Time</span>
                      </div>
                      <p className="text-xl font-bold text-white">{normalizeWeatherUnits(result.bestTime)}</p>
                    </GlassCard>
                  )}
                  
                  {result.avoidTime && (
                    <GlassCard className="p-4 border-red-500/20 bg-red-500/5">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-xs font-semibold text-red-400">Avoid</span>
                      </div>
                      <p className="text-xl font-bold text-white">{normalizeWeatherUnits(result.avoidTime)}</p>
                    </GlassCard>
                  )}
                </div>

                {/* Tips */}
                {result.tips && result.tips.length > 0 && (
                  <GlassCard className="p-5">
                    <h3 className="flex items-center gap-2 mb-4 text-sm font-semibold text-indigo-300">
                      <Sparkles className="w-4 h-4" />
                      Smart Tips
                    </h3>
                    <ul className="space-y-3">
                      {result.tips.map((tip, index) => (
                        <li key={index} className="flex items-start gap-3">
                          <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                            {index + 1}
                          </span>
                          <span className="text-sm text-gray-300">{normalizeWeatherUnits(tip)}</span>
                        </li>
                      ))}
                    </ul>
                  </GlassCard>
                )}

                {/* Current Weather Summary */}
                {weatherData && (
                  <GlassCard className="p-5">
                    <h3 className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-400">
                      <Sun className="w-4 h-4" />
                      Current Conditions
                    </h3>
                    <div className="grid grid-cols-2 gap-4 text-center sm:grid-cols-4">
                      <div>
                        <p className="text-xl font-bold text-white">{weatherData.current?.temp}°C</p>
                        <p className="text-[10px] text-gray-500">Temp</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold text-white">{weatherData.current?.humidity}%</p>
                        <p className="text-[10px] text-gray-500">Humidity</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold text-white">{weatherData.current?.wind} km/h</p>
                        <p className="text-[10px] text-gray-500">Wind km/h</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold text-white capitalize">{weatherData.current?.condition}</p>
                        <p className="text-[10px] text-gray-500">Condition</p>
                      </div>
                    </div>
                  </GlassCard>
                )}

                {/* Plan Again Button */}
                <button
                  onClick={resetPlan}
                  className="flex items-center justify-center w-full gap-2 py-3 font-medium text-white transition-all bg-gray-800 border border-gray-700 rounded-xl hover:bg-gray-700"
                >
                  <RefreshCw className="w-4 h-4" />
                  Plan Another Activity
                </button>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
