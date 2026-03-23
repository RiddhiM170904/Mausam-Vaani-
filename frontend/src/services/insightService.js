import { getPersonalizedInsight } from "../../AI/index.js";

const INSIGHT_CACHE_PREFIX = "mv_ai_quick_insight_v7";
const INSIGHT_LAST_ACTIVE_KEY = "mv_ai_last_active_at";
const INSIGHT_INACTIVITY_MS = 20 * 60 * 1000;
const QUICK_INSIGHT_MAX_CACHE_MS = 20 * 60 * 1000;
const AI_INSIGHT_DEBUG_PREFIX = "[AIInsightDebug]";
const MAX_INSIGHT_WORDS = 55;

function nowMs() {
  return Date.now();
}

function readCache(cacheKey) {
  try {
    const raw = localStorage.getItem(cacheKey);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function writeCache(cacheKey, payload) {
  try {
    localStorage.setItem(cacheKey, JSON.stringify(payload));
  } catch {
    // Ignore storage errors and continue without cache.
  }
}

function buildSessionInsightKey(data = {}) {
  const userId = data?.userId || data?.user_id || "guest";
  const lat = Number(data?.latitude ?? data?.location?.lat);
  const lon = Number(data?.longitude ?? data?.location?.lon);
  const persona = String(data?.persona || data?.user_profile?.user_type || "general").toLowerCase();
  const requirementsKey = String(data?.requirements || data?.notes || "")
    .trim()
    .toLowerCase()
    .slice(0, 80)
    .replace(/\s+/g, " ");
  const temp = Number(data?.weatherData?.current?.temp);
  const feelsLike = Number(data?.weatherData?.current?.feels_like ?? data?.weatherData?.current?.feelsLike);
  const rain = Number(data?.weatherData?.current?.rain_probability ?? data?.weatherData?.rain_probability);
  const humidity = Number(data?.weatherData?.current?.humidity);
  const wind = Number(data?.weatherData?.current?.wind);
  const condition = String(data?.weatherData?.current?.condition || '').toLowerCase();
  const tempBucket = Number.isFinite(temp) ? Math.round(temp / 2) : "na";
  const feelsBucket = Number.isFinite(feelsLike) ? Math.round(feelsLike / 2) : "na";
  const rainBucket = Number.isFinite(rain) ? Math.round(rain * 10) : "na";
  const humidityBucket = Number.isFinite(humidity) ? Math.round(humidity / 10) : "na";
  const windBucket = Number.isFinite(wind) ? Math.round(wind / 3) : "na";
  const latKey = Number.isFinite(lat) ? lat.toFixed(2) : "na";
  const lonKey = Number.isFinite(lon) ? lon.toFixed(2) : "na";
  const now = new Date();
  const llmProvider = String(import.meta.env.VITE_LLM_PROVIDER || "gemini").toLowerCase();
  const dayKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  const monthKey = now.toLocaleString('en-IN', { month: 'short' }).toLowerCase();
  const timeWindowKey = `${now.getHours()}-${Math.floor(now.getMinutes() / 20)}`;
  const conditionKey = condition.replace(/\s+/g, '_').slice(0, 24) || 'na';
  return `${INSIGHT_CACHE_PREFIX}:${llmProvider}:${userId}:${persona}:${latKey}:${lonKey}:${dayKey}:${monthKey}:w${timeWindowKey}:t${tempBucket}:f${feelsBucket}:r${rainBucket}:h${humidityBucket}:wd${windBucket}:c${conditionKey}:${requirementsKey}`;
}

function updateLastActive() {
  try {
    localStorage.setItem(INSIGHT_LAST_ACTIVE_KEY, String(nowMs()));
  } catch {
    // Ignore storage errors.
  }
}

function isSessionActive() {
  const raw = localStorage.getItem(INSIGHT_LAST_ACTIVE_KEY);
  const lastActive = Number(raw || 0);
  if (!Number.isFinite(lastActive) || !lastActive) {
    return false;
  }

  return nowMs() - lastActive <= INSIGHT_INACTIVITY_MS;
}

function toWords(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean);
}

function isIncompleteMessage(text) {
  const cleaned = String(text || "").trim();
  if (!cleaned) return true;

  const words = toWords(cleaned);
  if (words.length < 14) return true;

  if (/(\bke|\bki|\bka|\baur|\bto|\bpar|\bme|\bmein|\bya|\bwith|\band)[.!?]?$/i.test(cleaned)) {
    return true;
  }

  return false;
}

function enforceInsightWordRange(text, maxWords = MAX_INSIGHT_WORDS) {
  let message = String(text || "").replace(/\s+/g, " ").trim();

  if (!message) {
    message =
      "Aaj weather thoda changeable hai, isliye plan smart rakho. Hydration, light protection, aur travel timing pe focus karo. Agar conditions shift ho, short breaks lo aur next few hours ka forecast dobara check kar lena.";
  }

  const words = toWords(message);
  if (words.length > maxWords) {
    const connectors = new Set(["aur", "or", "to", "ke", "ki", "ka", "me", "mein", "par", "pe", "ya", "for", "and", "with", "the", "a", "an", "ko", "se", "hai"]);
    let clipped = words.slice(0, maxWords);

    while (clipped.length > 18 && connectors.has(String(clipped[clipped.length - 1] || "").toLowerCase())) {
      clipped.pop();
    }

    message = clipped.join(" ").replace(/[,:;\-]+$/, "").trim();
  }

  if (message && !/[.!?]$/.test(message)) {
    message = `${message}.`;
  }

  return message;
}

function normalizeInsightMessage(text) {
  let message = String(text || '').replace(/\s+/g, ' ').trim();

  // Home card already renders user greeting, so remove repeated opening greeting from model output.
  message = message
    .replace(/^hey\s+[a-zA-Z][a-zA-Z\s]{0,20}\s*👋\s*/i, '')
    .replace(/^hey\s*👋\s*/i, '')
    .replace(/^hi\s+[a-zA-Z][a-zA-Z\s]{0,20}\s*👋\s*/i, '')
    .replace(/^hi\s*👋\s*/i, '')
    .trim();

  return message;
}

function normalizeTipText(text, severity = 'medium') {
  const raw = String(text || '').replace(/\s+/g, ' ').trim();
  if (!raw) return '';

  if (/moderate air quality/i.test(raw)) {
    return '';
  }

  if (/unhealthy air quality|severe air quality/i.test(raw)) {
    return '';
  }

  if (/high temperature conditions/i.test(raw)) {
    return 'Dhoop tez ho sakti hai; paani saath rakho aur 12-4 PM heavy outdoor kaam avoid karo.';
  }

  if (/heavy rainfall likely/i.test(raw)) {
    return 'Baarish ka chance strong hai; umbrella ya raincoat ready rakho.';
  }

  if (/strong winds expected/i.test(raw)) {
    return 'Hawa tez ho sakti hai; travel me speed controlled rakho.';
  }

  if (severity === 'low' && raw.split(' ').length <= 3) {
    return '';
  }

  return raw;
}

function buildActionableTips(aiData) {
  const tips = [];
  const notifications = Array.isArray(aiData?.notifications) ? aiData.notifications : [];
  const rag = Array.isArray(aiData?.rag_context) ? aiData.rag_context : [];

  notifications.forEach((item) => {
    const text = String(item?.message || '').trim();
    if (text) tips.push(text);
  });

  rag.forEach((item) => {
    const tip = normalizeTipText(item?.text, item?.severity || 'medium');
    if (tip) tips.push(tip);
  });

  return [...new Set(tips)].slice(0, 5);
}

function inferRiskLevelFromWeather(weatherData = {}, risks = []) {
  const current = weatherData?.current || {};
  const temp = Number(current?.temp || 0);
  const wind = Number(current?.wind || 0);
  const condition = String(current?.condition || '').toLowerCase();

  let score = 0;
  if (temp >= 40) score += 3;
  else if (temp >= 35) score += 2;
  else if (temp >= 31) score += 1;

  if (condition.includes('rain') || condition.includes('storm') || condition.includes('thunder')) score += 2;
  if (condition.includes('fog') || condition.includes('mist')) score += 1;
  if (wind >= 25) score += 2;
  else if (wind >= 16) score += 1;

  if (Array.isArray(risks) && risks.length > 0) score += 1;

  if (score >= 4) return 'High';
  if (score >= 2) return 'Medium';
  return 'Low';
}

function extractPlannerWindows(text = '') {
  const content = String(text || '');
  const timeRangeMatch = content.match(/(\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\s?(?:-|to)\s?\d{1,2}(?::\d{2})?\s?(?:am|pm)\b)/i);
  const singleTimeMatch = content.match(/\b(\d{1,2}(?::\d{2})?\s?(?:am|pm))\b/i);
  const twentyFourRangeMatch = content.match(/\b([01]?\d|2[0-3]):[0-5]\d\s?(?:-|to)\s?([01]?\d|2[0-3]):[0-5]\d\b/i);
  const twentyFourSingleMatch = content.match(/\b([01]?\d|2[0-3]):[0-5]\d\b/);

  return {
    bestTime: timeRangeMatch?.[1] || singleTimeMatch?.[1] || twentyFourRangeMatch?.[0] || twentyFourSingleMatch?.[0] || null,
    avoidTime: /avoid|mat|stay indoor|indoors|peak heat|loo/i.test(content)
      ? (timeRangeMatch?.[1] || twentyFourRangeMatch?.[0] || null)
      : null,
  };
}

function buildPlannerRequirements(plannerData = {}) {
  const activity = plannerData?.activity || 'daily routine';
  const date = plannerData?.date || 'today';
  const start = plannerData?.timeRange?.start || '09:00';
  const end = plannerData?.timeRange?.end || '18:00';
  const duration = plannerData?.duration || 'not specified';
  const risks = Array.isArray(plannerData?.risks) && plannerData.risks.length
    ? plannerData.risks.join(', ')
    : 'none';
  const notes = plannerData?.notes || 'none';

  return [
    'Planner mode enabled.',
    `User activity: ${activity}.`,
    `Date: ${date}.`,
    `Time range: ${start} to ${end}.`,
    `Duration preference: ${duration}.`,
    `Risk priorities: ${risks}.`,
    `User notes: ${notes}.`,
    'Give recommendation in Hinglish with practical timing guidance and safety actions.',
    'Mention best time window and avoid window naturally in the response if risk exists.',
  ].join(' ');
}

/**
 * AI Insight service — provides quick insights and comprehensive planning.
 * Two types:
 * 1. Quick Insight (Dashboard) - fast, lightweight advice
 * 2. Smart Planner - comprehensive scenario-based predictions
 * 
 * Uses frontend AI pipeline (weather + RAG + Gemini) for insights
 */
export const insightService = {
  /**
   * Quick insight for dashboard - lightweight, fast
   * Returns micro-advice like "Carry umbrella", "Avoid 2-4pm heat"
   * @param {Object} data - Object containing weatherData, persona, location, weatherRisks
   */
  async getQuickInsight(data) {
    const cacheKey = buildSessionInsightKey(data);
    const cached = readCache(cacheKey);
    const cachedMessage = cached?.value?.message;
    const cachedSource = String(cached?.value?.source || cached?.value?.llm_source || "").toLowerCase();
    const isTrustedLlmSource = cachedSource.startsWith("groq:") || cachedSource.startsWith("gemini:");
    const cacheAge = nowMs() - Number(cached?.savedAt || 0);
    const canUseCached =
      cached &&
      isTrustedLlmSource &&
      cacheAge <= QUICK_INSIGHT_MAX_CACHE_MS &&
      isSessionActive() &&
      !isIncompleteMessage(cachedMessage);

    if (canUseCached) {
      if (cached?.value && typeof cached.value === "object") {
        cached.value.message = normalizeInsightMessage(enforceInsightWordRange(cached?.value?.message));
        if (Array.isArray(cached.value.tips)) {
          cached.value.tips = cached.value.tips
            .map((tip) => normalizeTipText(tip, 'medium'))
            .filter(Boolean)
            .slice(0, 5);
        }
        cached.value.generatedAt = cached.value.generatedAt || cached?.savedAt || nowMs();
      }
      console.info(`${AI_INSIGHT_DEBUG_PREFIX} cache hit`, {
        cacheKey,
        source: cached?.value?.source || cached?.value?.llm_source || "cache",
      });
      updateLastActive();
      return cached.value;
    }

    const { weatherData, persona = 'general', location = null, weatherRisks = [] } = data || {};
    const lat = data?.latitude ?? location?.lat ?? null;
    const lon = data?.longitude ?? location?.lon ?? null;
    const locationName = data?.location_name || location?.city || location?.formattedAddress || 'Unknown';

    console.info(`${AI_INSIGHT_DEBUG_PREFIX} request start`, {
      cacheKey,
      userId: data?.userId || data?.user_id || null,
      persona,
      lat,
      lon,
      locationName,
      hasRequirements: Boolean(data?.requirements),
    });

    try {
      const aiPayload = {
        user_id: data?.userId || data?.user_id || null,
        location: {
          lat,
          lon,
        },
        user_profile: {
          user_id: data?.userId || data?.user_id || null,
          user_type: persona,
          location: {
            lat,
            lon,
            city: locationName,
          },
          profile: {
            weather_risks: weatherRisks,
            planner_answers: data?.plannerProfile?.answers || {},
          },
        },
        requirements:
          data?.requirements ||
          data?.notes ||
          "Provide short, actionable weather guidance for this user.",
      };

      const aiResult = await getPersonalizedInsight(aiPayload);
      const payload = {
        success: true,
        title: "AI Insight",
        message: normalizeInsightMessage(enforceInsightWordRange(aiResult?.data?.insight || "No insight available yet.")),
        tips: buildActionableTips(aiResult?.data),
        source: aiResult?.data?.llm_source || "fallback",
        ai: aiResult?.data || null,
        generatedAt: nowMs(),
      };

      writeCache(cacheKey, {
        value: payload,
        savedAt: nowMs(),
      });
      updateLastActive();

      console.info(`${AI_INSIGHT_DEBUG_PREFIX} response`, {
        cacheKey,
        source: payload.source,
        messagePreview: String(payload.message || "").slice(0, 120),
      });

      return payload;
    } catch (error) {
      console.warn(`${AI_INSIGHT_DEBUG_PREFIX} quick-insight failed`, {
        cacheKey,
        error: error.message,
      });

      const fallback = this.generateLocalInsight(weatherData, persona, weatherRisks);
      writeCache(cacheKey, {
        value: fallback,
        savedAt: nowMs(),
      });
      updateLastActive();
      return fallback;
    }
  },

  /**
   * Comprehensive AI planner - scenario-based predictions
   * Takes activity, date, time range, risks, and returns smart recommendations
   * Calls FastAPI AI-Backend with Gemini integration
   */
  async getSmartPlan(plannerData) {
    console.info(`${AI_INSIGHT_DEBUG_PREFIX} planner request start`, {
      activity: plannerData?.activity,
      persona: plannerData?.persona,
    });

    try {
      const location = plannerData?.location || {};
      const lat = plannerData?.latitude ?? location?.lat ?? null;
      const lon = plannerData?.longitude ?? location?.lon ?? null;

      const aiPayload = {
        intent: 'planner',
        user_id: plannerData?.userId || plannerData?.user_id || null,
        location: { lat, lon },
        planner_context: {
          activity: plannerData?.activity || 'daily_routine',
          date: plannerData?.date || 'today',
          time_range: plannerData?.timeRange || { start: '09:00', end: '18:00' },
          time_preset: plannerData?.timePreset || 'custom',
          duration: plannerData?.duration || null,
          risks: plannerData?.risks || [],
          notes: plannerData?.notes || '',
        },
        user_profile: {
          user_id: plannerData?.userId || plannerData?.user_id || null,
          user_type: plannerData?.persona || 'general',
          location: {
            lat,
            lon,
            city: plannerData?.location?.city || plannerData?.location_name || 'Unknown',
          },
          profile: {
            weather_risks: plannerData?.risks || [],
            planner_answers: plannerData?.plannerProfile?.answers || {},
          },
        },
        requirements: buildPlannerRequirements(plannerData),
      };

      const aiResult = await getPersonalizedInsight(aiPayload);
      const recommendationRaw = aiResult?.data?.insight || 'No planner recommendation available yet.';
      const recommendation = enforceInsightWordRange(recommendationRaw, 95);
      const windows = extractPlannerWindows(recommendationRaw);
      const riskLevel = inferRiskLevelFromWeather(plannerData?.weatherData, plannerData?.risks || []);

      const tips = buildActionableTips(aiResult?.data);

      return {
        success: true,
        recommendation,
        bestTime: windows.bestTime || `${plannerData?.timeRange?.start || '09:00'}-${plannerData?.timeRange?.end || '18:00'}`,
        avoidTime: windows.avoidTime || null,
        riskLevel,
        tips,
        activity: plannerData?.activity,
        source: aiResult?.data?.llm_source || 'ai',
        ai: aiResult?.data || null,
      };
    } catch (error) {
      console.warn(`${AI_INSIGHT_DEBUG_PREFIX} planner ai failed`, {
        activity: plannerData?.activity,
        error: error?.message,
      });
      return this.generateLocalPlan(plannerData);
    }
  },

  /**
   * Legacy method for backward compatibility
   */
  async getInsight(weatherData, persona = {}) {
    return this.getQuickInsight(weatherData, persona?.persona || 'general');
  },

  /**
   * Legacy planner advice method
   */
  async getPlannerAdvice(weatherData, date, activity) {
    return this.getSmartPlan({
      weatherData,
      date,
      activity,
      timeRange: { start: '09:00', end: '18:00' }
    });
  },

  /**
   * Generate local insight when API is unavailable
   * Rule-based smart advice based on weather conditions
   */
  generateLocalInsight(weather, persona, weatherRisks = []) {
    const insights = [];
    const current = weather?.current || {};
    const temp = current.temp || 25;
    const humidity = current.humidity || 50;
    const condition = (current.condition || '').toLowerCase();
    const wind = current.wind || 0;
    
    let title = 'Smart Advice';
    let primaryMessage = '';

    // Temperature-based advice
    if (temp >= 35) {
      primaryMessage = "Extreme heat! Stay hydrated and avoid outdoor activities 12-4pm";
      title = "Heat Alert";
      insights.push("🔥 " + primaryMessage);
    } else if (temp >= 30) {
      insights.push("☀️ High temperature. Carry water and wear light clothes");
    } else if (temp <= 15) {
      insights.push("🧥 Cold weather. Layer up before heading out");
    }

    // Rain advice
    if (condition.includes('rain') || condition.includes('drizzle')) {
      primaryMessage = primaryMessage || "Carry umbrella. Roads may be slippery";
      title = condition.includes('heavy') ? "Heavy Rain" : "Rain Expected";
      insights.push("🌧️ " + primaryMessage);
    }

    // Humidity advice
    if (humidity >= 80) {
      insights.push("💧 High humidity. Take breaks if working outdoors");
    }

    // Wind advice
    if (wind >= 20) {
      insights.push("💨 Strong winds expected. Secure loose items");
      if (!primaryMessage) {
        primaryMessage = "Strong winds expected. Secure loose items";
        title = "Wind Advisory";
      }
    }

    // Fog advice
    if (condition.includes('fog') || condition.includes('mist')) {
      primaryMessage = "Low visibility. Drive carefully";
      title = "Fog Alert";
      insights.push("🌫️ " + primaryMessage);
    }

    // Persona-specific advice
    if (persona === 'driver' || persona === 'delivery') {
      if (condition.includes('rain')) {
        insights.push("🚗 Increase following distance on wet roads");
      }
    } else if (persona === 'farmer') {
      if (condition.includes('rain')) {
        insights.push("🌾 Good day for field irrigation. Postpone fertilizer application");
      }
    } else if (persona === 'worker') {
      if (temp >= 32) {
        insights.push("👷 Take frequent breaks in shade. Hydrate every 30 mins");
      }
    }

    // Best time suggestion
    const hour = new Date().getHours();
    if (temp >= 30 && hour < 10) {
      insights.push("⏰ Best outdoor window: Now until 10 AM");
    } else if (temp >= 30) {
      insights.push("⏰ Better conditions expected after 5 PM");
    }

    // Set default message if none set
    if (!primaryMessage) {
      primaryMessage = insights[0]?.replace(/^[^\s]+\s/, '') || "Good weather conditions for your activities today.";
      title = 'Today\'s Tip';
    }

    return {
      success: true,
      title: title,
      message: enforceInsightWordRange(primaryMessage),
      tips: insights,
      source: 'local',
      generatedAt: nowMs(),
    };
  },

  /**
   * Generate local plan when API is unavailable
   */
  generateLocalPlan(plannerData) {
    const { weatherData, activity, date, timeRange, risks = [] } = plannerData;
    const current = weatherData?.current || {};
    const temp = current.temp || 25;
    const condition = (current.condition || '').toLowerCase();
    const hourly = weatherData?.hourly || [];

    const tips = [];
    let riskLevel = 'Low';
    let recommendation = 'Good conditions for your activity';
    let bestTime = timeRange?.start || '09:00';
    let avoidTime = null;

    // Activity-specific analysis
    if (activity === 'travel' || activity === 'commute') {
      if (condition.includes('rain')) {
        tips.push("Allow extra travel time due to wet roads");
        tips.push("Check traffic updates before departure");
        riskLevel = 'Medium';
      }
      if (temp >= 35) {
        tips.push("Ensure AC is working. Carry water");
        recommendation = "Avoid peak afternoon hours (12-4 PM)";
        avoidTime = '12:00 - 16:00';
      }
    }

    if (activity === 'outdoor' || activity === 'exercise') {
      if (temp >= 32) {
        recommendation = "Schedule for early morning (6-9 AM) or evening (5-7 PM)";
        bestTime = '06:00';
        avoidTime = '11:00 - 16:00';
        tips.push("Carry water and electrolytes");
        tips.push("Wear light, breathable clothing");
        riskLevel = 'Medium';
      }
      if (condition.includes('rain')) {
        tips.push("Consider indoor alternatives");
        riskLevel = 'High';
      }
    }

    if (activity === 'farming') {
      if (condition.includes('rain')) {
        recommendation = "Good day for indoor farm work. Avoid field operations";
        tips.push("Postpone pesticide and fertilizer application");
        tips.push("Check drainage systems");
      }
      if (temp >= 35) {
        recommendation = "Work in early morning (5-10 AM) or late evening";
        bestTime = '05:00';
        avoidTime = '11:00 - 17:00';
        tips.push("Hydrate workers frequently");
        tips.push("Provide shade breaks");
      }
    }

    if (activity === 'event') {
      if (condition.includes('rain')) {
        riskLevel = 'High';
        recommendation = "Have backup indoor venue ready";
        tips.push("Arrange canopy/tent coverage");
        tips.push("Prepare rain contingency plan");
      }
      if (temp >= 35) {
        tips.push("Arrange shade and cooling stations");
        tips.push("Provide water stations for guests");
      }
    }

    if (activity === 'delivery') {
      if (condition.includes('rain')) {
        tips.push("Protect packages from water damage");
        tips.push("Allow extra delivery time");
        riskLevel = 'Medium';
      }
      if (temp >= 35) {
        tips.push("Prioritize AC in vehicle during peak hours");
        tips.push("Take breaks in shade");
      }
    }

    // Risk-based adjustments
    if (risks.includes('avoid_rain') && condition.includes('rain')) {
      riskLevel = 'High';
      recommendation = "High rain probability. Consider postponing outdoor activities";
    }
    if (risks.includes('avoid_heat') && temp >= 32) {
      riskLevel = riskLevel === 'High' ? 'High' : 'Medium';
      recommendation = "High temperature expected. Best hours: before 10 AM or after 5 PM";
    }

    // Default tips if none generated
    if (tips.length === 0) {
      tips.push("Weather conditions are favorable");
      tips.push("Stay aware of any changes in forecast");
    }

    return {
      success: true,
      recommendation,
      bestTime,
      avoidTime,
      riskLevel,
      tips,
      activity,
      source: 'local'
    };
  }
};
