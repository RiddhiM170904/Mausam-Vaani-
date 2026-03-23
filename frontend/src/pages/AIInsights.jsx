import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, TriangleAlert, CalendarCheck2, ShieldCheck, Wind, Thermometer, CloudRain, CheckCircle2, XCircle } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import useLocation from "../hooks/useLocation";
import useWeather from "../hooks/useWeather";
import { insightService } from "../services/insightService";
import GlassCard from "../components/GlassCard";
import Loader from "../components/Loader";

function getSafeInsightMessage(insight) {
  if (!insight) return "No insight available yet.";
  if (typeof insight === "string") return insight;
  if (typeof insight?.message === "string" && insight.message.trim()) return insight.message;
  if (typeof insight?.recommendation === "string" && insight.recommendation.trim()) return insight.recommendation;
  if (Array.isArray(insight?.tips) && insight.tips.length > 0) return String(insight.tips[0]);
  return "No insight available yet.";
}

function buildActionItems(weather) {
  const current = weather?.current || {};
  const temp = Number(current.temp || 0);
  const humidity = Number(current.humidity || 0);
  const wind = Number(current.wind || 0);
  const condition = String(current.condition || "").toLowerCase();

  const actions = [];

  if (temp >= 35) actions.push("Avoid heavy outdoor activity between 12 PM and 4 PM.");
  if (condition.includes("rain")) actions.push("Keep rain protection ready and start commute earlier.");
  if (wind >= 20) actions.push("Expect stronger winds. Secure loose items and drive cautiously.");
  if (humidity >= 80) actions.push("Plan hydration breaks if spending long time outdoors.");
  if (!actions.length) actions.push("Weather looks stable. Continue with normal plans and monitor updates.");

  return actions;
}

function getCurrentHour() {
  return new Date().getHours();
}

function getTimeSlot(hour = getCurrentHour()) {
  if (hour < 5) return "night";
  if (hour < 12) return "morning";
  if (hour < 17) return "afternoon";
  if (hour < 21) return "evening";
  return "night";
}

function normalizeRainRisk(weatherData) {
  const current = weatherData?.current || {};
  const fromCurrent = Number(current?.rainProbability ?? current?.rain_probability);
  const fromHourly = Number(weatherData?.hourly?.[0]?.rainProbability ?? weatherData?.hourly?.[0]?.pop);
  const raw = Number.isFinite(fromCurrent) ? fromCurrent : fromHourly;
  if (!Number.isFinite(raw)) return 0;
  return raw > 1 ? raw / 100 : raw;
}

function riskBadge(level) {
  if (level === "High") return "🔴";
  if (level === "Moderate") return "🟡";
  return "🟢";
}

function getHeatRisk(temp) {
  if (temp >= 38) return "High";
  if (temp >= 33) return "Moderate";
  return "Low";
}

function getAqiRisk(aqi) {
  if (aqi >= 180) return "High";
  if (aqi >= 110) return "Moderate";
  return "Low";
}

function getRainRisk(rainProb) {
  if (rainProb >= 0.6) return "High";
  if (rainProb >= 0.35) return "Moderate";
  return "Low";
}

function buildSmartSummary(weatherData) {
  const temp = Number(weatherData?.current?.temp || 0);
  const aqi = Number(weatherData?.current?.aqi || 0);
  const rain = normalizeRainRisk(weatherData);

  if (temp >= 34 && aqi >= 130) return "Today: Warm with heavier air - avoid long outdoor exposure";
  if (rain >= 0.55) return "Today: Rain chance high - plan commute with buffer";
  if (temp >= 34) return "Today: Warm but manageable - avoid peak heat window";
  if (aqi >= 130) return "Today: Air quality moderate-high - reduce long outdoor stay";
  return "Today: Weather mostly comfortable - continue with smart timing";
}

function looksGenericInsight(text) {
  const value = String(text || "").toLowerCase();
  return /routine|balanced|stable|normal|manageable|check weather|stay updated/.test(value);
}

function buildCuratedInsight(weatherData, userName) {
  const temp = Number(weatherData?.current?.temp || 0);
  const aqi = Number(weatherData?.current?.aqi || 0);
  const rain = normalizeRainRisk(weatherData);
  const slot = getTimeSlot();
  const name = userName || "friend";

  if (rain >= 0.55) {
    return `Hey ${name} 👋 ${slot === "evening" ? "shaam ko weather thoda change ho sakta hai" : "aaj baarish ka chance zyada hai"} 🌧️ Bahar nikalte time umbrella le jaana aur travel me thoda buffer rakhna 👍`;
  }

  if (temp >= 34 && aqi >= 130) {
    return `Hey ${name} 👋 aaj heat ke saath hawa bhi thodi heavy lag sakti hai 🌫️ Bahar jao toh hydration aur mask ka dhyan rakhna, long outdoor stay avoid karo 👍`;
  }

  if (temp >= 34) {
    return `Hey ${name} 👋 aaj ${slot} me heat thodi zyada rahegi ☀️ Bahar jao toh paani saath rakho aur direct dhoop 12-4 PM avoid karna better rahega 👍`;
  }

  if (aqi >= 130) {
    return `Hey ${name} 👋 aaj air quality thodi heavy feel ho sakti hai 🌫️ Bahar nikalte waqt mask carry karo aur unnecessary outdoor wait avoid karo 👍`;
  }

  return `Hey ${name} 👋 aaj weather kaafi manageable lag raha hai 🙂 Outdoor plans theek hain, bas hydration maintain rakhna aur timing smart choose karna 👍`;
}

function buildInsightReasons(weatherData) {
  const temp = Number(weatherData?.current?.temp || 0);
  const rain = normalizeRainRisk(weatherData);
  const aqi = Number(weatherData?.current?.aqi || 0);
  const slot = getTimeSlot();
  const reasons = [];

  if (temp >= 33) reasons.push(`High temperature (${Math.round(temp)}°C)`);
  if (rain >= 0.35) reasons.push(`Rain probability elevated (${Math.round(rain * 100)}%)`);
  if (aqi >= 110) reasons.push(`Air quality impact (AQI ${Math.round(aqi)})`);
  reasons.push(`${slot.charAt(0).toUpperCase() + slot.slice(1)} weather window`);

  return reasons.slice(0, 3);
}

function parseTimeWindow(bestTime, weatherData) {
  const best = String(bestTime || '').trim();
  const twentyFourSingle = best.match(/^([01]?\d|2[0-3]):([0-5]\d)$/);
  const twentyFourRange = best.match(/^([01]?\d|2[0-3]):([0-5]\d)\s?(?:-|to)\s?([01]?\d|2[0-3]):([0-5]\d)$/i);

  if (twentyFourRange) {
    return {
      range: `${twentyFourRange[1].padStart(2, "0")}:${twentyFourRange[2]}-${twentyFourRange[3].padStart(2, "0")}:${twentyFourRange[4]}`,
      reason: "Lower-risk window from planner analysis for your current context.",
    };
  }

  if (twentyFourSingle) {
    const hour = Number(twentyFourSingle[1]);
    const start = `${String(hour).padStart(2, "0")}:${twentyFourSingle[2]}`;
    const end = `${String((hour + 2) % 24).padStart(2, "0")}:${twentyFourSingle[2]}`;
    return {
      range: `${start}-${end}`,
      reason: "Recommended short window around the planner suggestion time.",
    };
  }

  if (best && /\d/.test(best)) {
    return {
      range: best,
      reason: "Lower risk window based on activity and current weather context.",
    };
  }

  const temp = Number(weatherData?.current?.temp || 0);
  const rain = normalizeRainRisk(weatherData);

  if (rain >= 0.55) {
    return { range: "After rain breaks", reason: "Rain risk is high right now; wait for a drier slot." };
  }
  if (temp >= 34) {
    return { range: "6-8 PM", reason: "Evening is cooler and more comfortable than peak afternoon." };
  }

  return { range: "Morning or evening", reason: "Generally safer windows for outdoor tasks." };
}

function buildCarryItems(weatherData) {
  const temp = Number(weatherData?.current?.temp || 0);
  const rain = normalizeRainRisk(weatherData);
  const aqi = Number(weatherData?.current?.aqi || 0);
  const items = [];

  if (temp >= 33) {
    items.push("Water bottle 💧");
    items.push("Sunglasses 🕶️");
    items.push("Light cap 🧢");
  }
  if (rain >= 0.4) items.push("Umbrella / rain cover 🌧️");
  if (aqi >= 130) items.push("Mask 😷");
  if (!items.length) items.push("Water bottle 💧");

  return [...new Set(items)].slice(0, 4);
}

function buildDayPlan(weatherData) {
  const temp = Number(weatherData?.current?.temp || 0);
  const rain = normalizeRainRisk(weatherData);

  const morning = rain >= 0.5
    ? "Morning: Travel only if needed, keep rain protection ready."
    : "Morning: Good window for outdoor starts and commuting.";

  const afternoon = temp >= 34
    ? "Afternoon: Avoid long direct sun exposure between 12-4 PM."
    : "Afternoon: Keep hydration steady during active hours.";

  const evening = rain >= 0.5
    ? "Evening: Keep buffer in travel timing due to weather changes."
    : "Evening: Better window for activity and routine travel.";

  return [morning, afternoon, evening];
}

function buildDoAvoid(weatherData) {
  const temp = Number(weatherData?.current?.temp || 0);
  const rain = normalizeRainRisk(weatherData);
  const aqi = Number(weatherData?.current?.aqi || 0);

  const doList = ["Carry water", "Plan key travel in lower-risk window"];
  const avoidList = [];

  if (temp >= 34) avoidList.push("Direct sun exposure 12-4 PM");
  if (rain >= 0.5) avoidList.push("Leaving without rain protection");
  if (aqi >= 130) avoidList.push("Long outdoor exposure without mask");
  if (!avoidList.length) avoidList.push("Ignoring hydration during active hours");

  return { doList: doList.slice(0, 3), avoidList: avoidList.slice(0, 3) };
}

function computeAiConfidence(weatherData, plannerPlan) {
  const rain = normalizeRainRisk(weatherData);
  const temp = Number(weatherData?.current?.temp || 0);
  const hasPlanner = Boolean(plannerPlan?.recommendation);
  const hasWindow = Boolean(plannerPlan?.bestTime);

  let score = 68;
  if (temp >= 34 || rain >= 0.45) score += 7;
  if (hasPlanner) score += 8;
  if (hasWindow) score += 6;
  return Math.min(95, Math.max(70, score));
}

function buildProfileRequirements(user) {
  const persona = String(user?.persona || "general");
  const risks = Array.isArray(user?.weather_risks) ? user.weather_risks.join(", ") : "none";
  return [
    `User persona: ${persona}.`,
    `Risk preferences: ${risks}.`,
    "Generate clear daily decisions, timing guidance, and actionable steps.",
    "Avoid generic statements.",
  ].join(" ");
}

export default function AIInsights() {
  const { user, isLoggedIn } = useAuth();
  const { location } = useLocation();
  const { data: weatherData, loading: weatherLoading } = useWeather(location?.lat, location?.lon);
  const [insightLoading, setInsightLoading] = useState(false);
  const [insight, setInsight] = useState(null);
  const [plannerPlan, setPlannerPlan] = useState(null);

  useEffect(() => {
    if (!weatherData) return;

    setInsightLoading(true);
    insightService
      .getQuickInsight({
        userId: user?.id || null,
        weatherData,
        persona: user?.persona || "general",
        weatherRisks: user?.weather_risks || [],
        location,
        requirements: buildProfileRequirements(user),
      })
      .then((res) => setInsight(res))
      .finally(() => setInsightLoading(false));
  }, [weatherData, user?.id, user?.persona, user?.weather_risks, location?.lat, location?.lon, location?.city]);

  useEffect(() => {
    if (!weatherData) return;

    const now = new Date();
    const later = new Date(now.getTime() + 3 * 60 * 60 * 1000);
    const toHHmm = (value) => `${String(value.getHours()).padStart(2, '0')}:${String(value.getMinutes()).padStart(2, '0')}`;

    insightService
      .getSmartPlan({
        userId: user?.id || null,
        persona: user?.persona || 'general',
        location,
        weatherData,
        plannerProfile: user?.planner_profile || null,
        activity: 'daily_routine',
        date: now.toISOString().split('T')[0],
        timeRange: { start: toHHmm(now), end: toHHmm(later) },
        risks: user?.weather_risks || user?.weatherRisks || [],
        duration: 3,
        notes: `Generate practical short action plan in Hinglish. ${buildProfileRequirements(user)}`,
      })
      .then((plan) => setPlannerPlan(plan))
      .catch(() => setPlannerPlan(null));
  }, [weatherData, user?.id, user?.persona, user?.weather_risks, user?.weatherRisks, user?.planner_profile, location?.lat, location?.lon, location?.city]);

  const actionItems = useMemo(() => {
    if (Array.isArray(plannerPlan?.tips) && plannerPlan.tips.length > 0) {
      return plannerPlan.tips.slice(0, 5);
    }
    return buildActionItems(weatherData);
  }, [plannerPlan, weatherData]);

  const smartSummary = useMemo(() => buildSmartSummary(weatherData), [weatherData]);
  const baseInsight = getSafeInsightMessage(insight);
  const curatedInsight = useMemo(() => {
    const text = String(baseInsight || "");
    if (!text || looksGenericInsight(text)) {
      return buildCuratedInsight(weatherData, user?.name || user?.full_name || "friend");
    }
    return text;
  }, [baseInsight, weatherData, user?.name, user?.full_name]);
  const whyAdvice = useMemo(() => buildInsightReasons(weatherData), [weatherData]);

  const temp = Number(weatherData?.current?.temp || 0);
  const aqi = Number(weatherData?.current?.aqi || 0);
  const rainProb = normalizeRainRisk(weatherData);
  const heatRisk = getHeatRisk(temp);
  const aqiRisk = getAqiRisk(aqi);
  const rainRisk = getRainRisk(rainProb);

  const bestWindow = useMemo(() => parseTimeWindow(plannerPlan?.bestTime, weatherData), [plannerPlan?.bestTime, weatherData]);
  const carryItems = useMemo(() => buildCarryItems(weatherData), [weatherData]);
  const todayPlan = useMemo(() => buildDayPlan(weatherData), [weatherData]);
  const doAvoid = useMemo(() => buildDoAvoid(weatherData), [weatherData]);
  const confidence = useMemo(() => computeAiConfidence(weatherData, plannerPlan), [weatherData, plannerPlan]);

  const friendlyName = (user?.name || user?.full_name || "friend").trim();
  const personalizedIntro = isLoggedIn ? `Hey ${friendlyName} 👋` : "Hey there 👋";

  if (weatherLoading) return <Loader text="Generating AI insights..." />;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full max-w-4xl mx-auto space-y-4 sm:space-y-6">
      <div className="px-1 space-y-1">
        <h1 className="text-2xl font-bold text-white">AI Insights</h1>
        <p className="text-sm text-gray-400">
          Decision-focused weather intelligence for {isLoggedIn ? user?.name || "you" : "your day"}.
        </p>
        <p className="text-xs font-medium text-indigo-300">{smartSummary}</p>
      </div>

      <GlassCard className="p-5 border-indigo-500/20 bg-linear-to-r from-indigo-500/10 to-blue-500/10 sm:p-6">
        <div className="flex items-start gap-3 sm:gap-4">
          <div className="flex items-center justify-center w-10 h-10 shrink-0 rounded-xl bg-indigo-500/20">
            <Sparkles className="w-5 h-5 text-indigo-300" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold tracking-wide text-indigo-300 uppercase">Personalized Recommendation</p>
            <p className="mt-1 text-sm font-medium text-indigo-200">{personalizedIntro}</p>
            {insightLoading ? (
              <p className="mt-2 text-sm text-gray-400">Analyzing your weather and risk profile...</p>
            ) : (
              <p className="mt-2 text-sm leading-relaxed text-gray-200">{curatedInsight}</p>
            )}
            {!insightLoading && whyAdvice.length > 0 && (
              <div className="mt-3 pt-3 border-t border-white/10">
                <p className="text-[11px] font-semibold tracking-wide text-indigo-300 uppercase">Why this advice</p>
                <div className="mt-1 space-y-1">
                  {whyAdvice.map((reason, idx) => (
                    <p key={idx} className="text-xs text-gray-300">• {reason}</p>
                  ))}
                </div>
                <p className="mt-2 text-[11px] text-gray-400">AI Confidence: {confidence}%</p>
              </div>
            )}
          </div>
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <GlassCard className="p-4" hover={false}>
          <div className="flex items-center gap-2 mb-2 text-red-300">
            <TriangleAlert className="w-4 h-4" />
            <p className="text-xs font-semibold tracking-wide uppercase">Risk Snapshot</p>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-gray-300">Heat Risk: {riskBadge(heatRisk)} {heatRisk}</p>
            <p className="text-sm text-gray-300">AQI Risk: {riskBadge(aqiRisk)} {aqiRisk}</p>
            <p className="text-sm text-gray-300">Rain Risk: {riskBadge(rainRisk)} {rainRisk}</p>
          </div>
        </GlassCard>

        <GlassCard className="p-4" hover={false}>
          <div className="flex items-center gap-2 mb-2 text-emerald-300">
            <CalendarCheck2 className="w-4 h-4" />
            <p className="text-xs font-semibold tracking-wide uppercase">Best Window</p>
          </div>
          <p className="text-sm font-semibold text-gray-200">{bestWindow.range}</p>
          <p className="mt-1 text-sm text-gray-400">{bestWindow.reason}</p>
        </GlassCard>

        <GlassCard className="p-4" hover={false}>
          <div className="flex items-center gap-2 mb-2 text-cyan-300">
            <ShieldCheck className="w-4 h-4" />
            <p className="text-xs font-semibold tracking-wide uppercase">Preparedness</p>
          </div>
          <div className="space-y-1">
            {carryItems.map((item, idx) => (
              <p key={idx} className="text-sm text-gray-300">• {item}</p>
            ))}
          </div>
        </GlassCard>
      </div>

      <GlassCard className="p-5 sm:p-6" hover={false}>
        <h2 className="mb-4 text-sm font-semibold tracking-wide text-gray-300 uppercase">Action Plan For Today</h2>
        <div className="space-y-3">
          {(todayPlan.length ? todayPlan : actionItems).map((item, idx) => (
            <div key={idx} className="flex items-start gap-3 p-3 border rounded-xl border-white/10 bg-white/5">
              <span className="mt-0.5 text-indigo-300">•</span>
              <p className="text-sm text-gray-200">{item}</p>
            </div>
          ))}
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <GlassCard className="p-4" hover={false}>
          <div className="flex items-center gap-2 mb-2 text-emerald-300">
            <CheckCircle2 className="w-4 h-4" />
            <p className="text-xs font-semibold tracking-wide uppercase">Do</p>
          </div>
          <div className="space-y-1">
            {doAvoid.doList.map((item, idx) => (
              <p key={idx} className="text-sm text-gray-300">• {item}</p>
            ))}
          </div>
        </GlassCard>

        <GlassCard className="p-4" hover={false}>
          <div className="flex items-center gap-2 mb-2 text-red-300">
            <XCircle className="w-4 h-4" />
            <p className="text-xs font-semibold tracking-wide uppercase">Avoid</p>
          </div>
          <div className="space-y-1">
            {doAvoid.avoidList.map((item, idx) => (
              <p key={idx} className="text-sm text-gray-300">• {item}</p>
            ))}
          </div>
        </GlassCard>
      </div>

      <div className="grid grid-cols-3 gap-3 sm:gap-4">
        <GlassCard className="p-3 text-center" hover={false}>
          <CloudRain className="w-4 h-4 mx-auto text-blue-300" />
          <p className="mt-2 text-xs text-gray-400">Humidity</p>
          <p className="text-sm font-semibold text-white">{weatherData?.current?.humidity ?? "--"}%</p>
        </GlassCard>
        <GlassCard className="p-3 text-center" hover={false}>
          <Thermometer className="w-4 h-4 mx-auto text-orange-300" />
          <p className="mt-2 text-xs text-gray-400">Feels Like</p>
          <p className="text-sm font-semibold text-white">{weatherData?.current?.feelsLike ?? "--"}°</p>
        </GlassCard>
        <GlassCard className="p-3 text-center" hover={false}>
          <Wind className="w-4 h-4 mx-auto text-cyan-300" />
          <p className="mt-2 text-xs text-gray-400">Wind</p>
          <p className="text-sm font-semibold text-white">{weatherData?.current?.wind ?? "--"} km/h</p>
        </GlassCard>
      </div>
    </motion.div>
  );
}
