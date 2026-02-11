import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "../context/AuthContext";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import { insightService } from "../services/insightService";
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

export default function Planner() {
  const { isLoggedIn, user } = useAuth();
  const { location } = useLocation();
  const { data: weatherData, loading: weatherLoading } = useWeather(location?.lat, location?.lon);

  // Form state
  const [activity, setActivity] = useState("");
  const [customActivity, setCustomActivity] = useState("");
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [timeRange, setTimeRange] = useState({ start: "09:00", end: "18:00" });
  const [risks, setRisks] = useState([]);
  const [duration, setDuration] = useState(4);
  const [notes, setNotes] = useState("");
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showForm, setShowForm] = useState(true);

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

  // Generate smart plan
  const generatePlan = async () => {
    if (!activity) return;
    
    setLoading(true);
    setResult(null);
    
    try {
      const plannerData = {
        persona: user?.persona || 'general',
        location: location,
        weatherData: weatherData,
        activity: activity === 'other' ? customActivity : activity,
        date: selectedDate,
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

  // If not logged in, show auth prompt
  if (!isLoggedIn) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex flex-col items-center justify-center py-20 text-center"
      >
        <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6">
          <Sparkles className="w-10 h-10 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-3">AI Weather Planner</h2>
        <p className="text-gray-400 text-sm max-w-sm mb-6">
          Get personalized weather-smart recommendations for your activities.
          Sign in to access the AI Planner.
        </p>
        <a
          href="/login"
          className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl font-medium hover:from-indigo-600 hover:to-purple-700 transition-all flex items-center gap-2"
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
      className="space-y-6 max-w-2xl mx-auto"
    >
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Sparkles className="w-6 h-6 text-indigo-400" />
          <h1 className="text-2xl font-bold text-white">AI Weather Planner</h1>
        </div>
        <p className="text-gray-400 text-sm">Plan your day smarter with weather insights</p>
      </div>

      {/* User Context Card */}
      <GlassCard className="p-4">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Your Profile</h3>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="flex flex-col items-center gap-1">
            <User className="w-5 h-5 text-indigo-400" />
            <span className="text-xs text-gray-400">Persona</span>
            <span className="text-sm text-white font-medium">{getPersonaLabel(user?.persona)}</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <MapPin className="w-5 h-5 text-blue-400" />
            <span className="text-xs text-gray-400">Location</span>
            <span className="text-sm text-white font-medium truncate max-w-[100px]">{location?.city || 'Unknown'}</span>
          </div>
          <div className="flex flex-col items-center gap-1">
            <Thermometer className="w-5 h-5 text-orange-400" />
            <span className="text-xs text-gray-400">Now</span>
            <span className="text-sm text-white font-medium">{weatherData?.current?.temp || '--'}°C</span>
          </div>
        </div>
      </GlassCard>

      <AnimatePresence mode="wait">
        {showForm ? (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-5"
          >
            {/* Activity Selection */}
            <GlassCard className="p-5">
              <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
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
                    <span className="text-xl block mb-1">{act.emoji}</span>
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
                    className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
                  />
                </motion.div>
              )}
            </GlassCard>

            {/* Date and Time */}
            <GlassCard className="p-5">
              <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
                <Calendar className="w-4 h-4 text-indigo-400" />
                When?
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Date picker */}
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">Date</label>
                  <input
                    type="date"
                    value={selectedDate}
                    onChange={(e) => setSelectedDate(e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                    className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
                  />
                </div>
                
                {/* Start time */}
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">Start Time</label>
                  <input
                    type="time"
                    value={timeRange.start}
                    onChange={(e) => setTimeRange(prev => ({ ...prev, start: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
                  />
                </div>
                
                {/* End time */}
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">End Time</label>
                  <input
                    type="time"
                    value={timeRange.end}
                    onChange={(e) => setTimeRange(prev => ({ ...prev, end: e.target.value }))}
                    className="w-full px-3 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
                  />
                </div>
              </div>
            </GlassCard>

            {/* Risk Priorities */}
            <GlassCard className="p-5">
              <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-400" />
                What to avoid? <span className="text-xs text-gray-500 font-normal">(optional)</span>
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
              <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                Duration <span className="text-xs text-gray-500 font-normal">(optional)</span>
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
              <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                <MoreHorizontal className="w-4 h-4 text-indigo-400" />
                Additional Notes <span className="text-xs text-gray-500 font-normal">(optional)</span>
              </h3>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="E.g., Carrying equipment, elderly passengers, kids..."
                rows={2}
                className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm resize-none"
              />
            </GlassCard>

            {/* Generate Button */}
            <button
              onClick={generatePlan}
              disabled={!activity || loading || (activity === 'other' && !customActivity)}
              className="w-full py-4 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl font-semibold hover:from-indigo-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
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
                      <p className="text-white font-medium leading-relaxed">
                        {result.recommendation}
                      </p>
                    </div>
                  </div>
                </GlassCard>

                {/* Time Recommendations */}
                <div className="grid grid-cols-2 gap-4">
                  {result.bestTime && (
                    <GlassCard className="p-4 border-green-500/20 bg-green-500/5">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span className="text-xs text-green-400 font-semibold">Best Time</span>
                      </div>
                      <p className="text-xl font-bold text-white">{result.bestTime}</p>
                    </GlassCard>
                  )}
                  
                  {result.avoidTime && (
                    <GlassCard className="p-4 border-red-500/20 bg-red-500/5">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-xs text-red-400 font-semibold">Avoid</span>
                      </div>
                      <p className="text-xl font-bold text-white">{result.avoidTime}</p>
                    </GlassCard>
                  )}
                </div>

                {/* Tips */}
                {result.tips && result.tips.length > 0 && (
                  <GlassCard className="p-5">
                    <h3 className="text-sm font-semibold text-indigo-300 mb-4 flex items-center gap-2">
                      <Sparkles className="w-4 h-4" />
                      Smart Tips
                    </h3>
                    <ul className="space-y-3">
                      {result.tips.map((tip, index) => (
                        <li key={index} className="flex items-start gap-3">
                          <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                            {index + 1}
                          </span>
                          <span className="text-sm text-gray-300">{tip}</span>
                        </li>
                      ))}
                    </ul>
                  </GlassCard>
                )}

                {/* Current Weather Summary */}
                {weatherData && (
                  <GlassCard className="p-5">
                    <h3 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2">
                      <Sun className="w-4 h-4" />
                      Current Conditions
                    </h3>
                    <div className="grid grid-cols-4 gap-3 text-center">
                      <div>
                        <p className="text-xl font-bold text-white">{weatherData.current?.temp}°</p>
                        <p className="text-[10px] text-gray-500">Temp</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold text-white">{weatherData.current?.humidity}%</p>
                        <p className="text-[10px] text-gray-500">Humidity</p>
                      </div>
                      <div>
                        <p className="text-xl font-bold text-white">{weatherData.current?.wind}</p>
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
                  className="w-full py-3 bg-gray-800 text-white rounded-xl font-medium hover:bg-gray-700 transition-all flex items-center justify-center gap-2 border border-gray-700"
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
