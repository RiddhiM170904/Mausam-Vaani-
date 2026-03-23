import { AI_CONFIG, hasGeminiKey } from './aiConfig.js';

const insightCache = new Map();
const GEMINI_LAST_MODEL_KEY = 'mv_gemini_last_model';

const USER_TONE_GUIDANCE = {
  driver: 'Focus on commute safety, slippery roads, visibility, and timing suggestions.',
  delivery: 'Focus on delay buffers, route practicality, and hydration during full-day movement.',
  farmer: 'Focus on irrigation, crop protection, and practical field timing decisions.',
  worker: 'Focus on heat stress, AQI exposure, breaks, hydration, and safer work windows.',
  office: 'Focus on office commute timing, traffic risk, and rain preparedness.',
  office_employee: 'Focus on office commute timing, traffic risk, and rain preparedness.',
  student: 'Focus on school/college timing, rain protection, and safety reminders.',
  senior: 'Focus on comfort, low exposure, mask/hydration, and caution in extreme weather.',
  senior_citizen: 'Focus on comfort, low exposure, mask/hydration, and caution in extreme weather.',
  business_owner: 'Focus on footfall impact, operations planning, and weather readiness.',
  commuter: 'Focus on travel timing, traffic, exposure reduction, and practical carry items.',
  general: 'Give clear day-level guidance and practical actions in simple language.',
};

function toWords(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(Boolean);
}

function looksIncomplete(text) {
  const cleaned = String(text || '').trim();
  if (!cleaned) return true;

  const words = toWords(cleaned);
  if (words.length < 8) return true;

  // Reject endings like "26°C ke." or trailing connector words.
  if (/(\bke|\bki|\bka|\baur|\bto|\bpar|\bme|\bmein|\bya|\bwith|\band)[.!?]?$/i.test(cleaned)) {
    return true;
  }

  return false;
}

function getTimeSlotFromHour(hour) {
  if (hour < 5) return 'night';
  if (hour < 12) return 'morning';
  if (hour < 17) return 'afternoon';
  if (hour < 21) return 'evening';
  return 'night';
}

function toAreaType(locationContext) {
  const value = String(locationContext || '').toLowerCase();
  if (value.includes('rural') || value.includes('village')) return 'rural';
  return 'urban';
}

function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

function getTimeContext(hour) {
  const h = Number(hour);
  if (!Number.isFinite(h)) return 'day_transition';
  if (h < 5) return 'late_night';
  if (h < 8) return 'early_morning';
  if (h < 12) return 'morning_active';
  if (h < 16) return 'afternoon_peak';
  if (h < 20) return 'evening_transition';
  return 'night_settle';
}

function buildReasoningContext(context, ragContext) {
  const weather = context?.weather || {};
  const user = context?.user_profile || {};
  const localHour = Number(context?.local_hour ?? new Date().getHours());
  const season = String(context?.season_hint || 'summer').toLowerCase();
  const timeContext = getTimeContext(localHour);

  const temp = Number(weather?.temp || 0);
  const rainProb = Number(weather?.rain_probability || 0);
  const aqi = Number(weather?.aqi || 0);
  const humidity = Number(weather?.humidity || 0);
  const windSpeed = Number(weather?.wind_speed || 0);

  // Soft weighting signals for LLM reasoning; not hard branches.
  const timeMultipliers = {
    heat: timeContext === 'afternoon_peak' ? 1.2 : timeContext === 'morning_active' ? 1.05 : 1,
    rain: timeContext === 'evening_transition' ? 1.1 : 1,
    cold: timeContext === 'early_morning' || timeContext === 'late_night' ? 1.15 : 1,
    pollution: timeContext === 'morning_active' || timeContext === 'evening_transition' ? 1.08 : 1,
  };

  const seasonMultipliers = {
    heat: season === 'summer' ? 1.2 : 1,
    rain: season === 'monsoon' ? 1.25 : 1,
    cold: season === 'winter' ? 1.25 : 1,
    pollution: season === 'winter' ? 1.15 : 1,
  };

  const scores = {
    heat: clamp01((temp / 50) * timeMultipliers.heat * seasonMultipliers.heat),
    rain: clamp01(rainProb * timeMultipliers.rain * seasonMultipliers.rain),
    cold: clamp01(((18 - temp) / 20) * timeMultipliers.cold * seasonMultipliers.cold),
    pollution: clamp01((aqi / 300) * timeMultipliers.pollution * seasonMultipliers.pollution),
    humidity: clamp01(humidity / 100),
    wind: clamp01(windSpeed / 25),
  };

  const sortedSignals = Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .map(([name, score]) => ({ name, score: Number(score.toFixed(3)) }));

  return {
    weather: {
      temperature_level: temp,
      rain_level: rainProb,
      aqi_level: aqi,
      humidity_level: humidity,
      wind_level: windSpeed,
    },
    context: {
      time_context: timeContext,
      time_slot: getTimeSlotFromHour(localHour),
      local_hour: localHour,
      month: String(context?.month_name || new Date().toLocaleString('en-IN', { month: 'long' })),
      season_context: season,
      area_type: toAreaType(context?.location_context),
      user_type: String(user?.user_type || 'general').toLowerCase(),
      user_profile: user?.profile || {},
    },
    combined_risk_score: scores,
    top_signals: sortedSignals.slice(0, 3),
    rag: (ragContext || []).map((item) => ({
      id: item?.id,
      severity: item?.severity || 'medium',
      text: item?.text || '',
    })),
  };
}

function sanitizeInsightOutput(text) {
  let cleaned = String(text || '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/\r/g, '')
    .trim();

  // Remove duplicate greeting lines but keep the first greeting.
  cleaned = cleaned.replace(/^(hey.*?\n)+/i, (match) => {
    const first = match.split('\n')[0];
    return `${first} `;
  });

  // Remove RAG leakage patterns such as "(Moderate air quality)".
  cleaned = cleaned.replace(/\(.*?air quality.*?\)/gi, '');

  cleaned = cleaned
    .replace(/^friendly message\s*[:\-]?\s*/i, '')
    .replace(/^output\s*[:\-]?\s*/i, '')
    .trim();

  // Remove bullet prefixes if model emits list output.
  cleaned = cleaned
    .replace(/^[-*•]\s*/gm, '')
    .replace(/check weather.*?(\.|$)/gi, '')
    .replace(/routine continue.*?(\.|$)/gi, '')
    .trim();

  const lines = cleaned
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 3);

  return lines.join(' ').replace(/\s+/g, ' ').trim();
}

function seemsActionable(text) {
  const msg = String(text || '').toLowerCase();
  if (!msg) return false;
  return /(carry|avoid|use|wear|take|keep|plan|check|drink|niklo|le\s*jana|pehno|rakho|bacho|hydrate|mask)/i.test(msg);
}

function isGeneric(text) {
  return /routine|check weather|stay updated|normal rakh sakte ho/i.test(String(text || ''));
}

function buildCacheKey(context, ragContext) {
  const hour = Number(context?.local_hour ?? new Date().getHours());
  const timeWindow = Math.floor(hour / 3);
  const monthName = String(context?.month_name || new Date().toLocaleString('en-IN', { month: 'long' }));
  const profile = context?.user_profile?.profile || {};
  const intent = String(context?.intent || 'insight').toLowerCase();
  const plannerContext = context?.planner_context || {};
  const requirements = String(context?.requirements || '')
    .trim()
    .toLowerCase()
    .slice(0, 120);

  return JSON.stringify({
    userType: context?.user_profile?.user_type || 'general',
    vehicle: context?.user_profile?.profile?.vehicle || '',
    intent,
    plannerActivity: String(plannerContext?.activity || ''),
    plannerDate: String(plannerContext?.date || ''),
    plannerTimePreset: String(plannerContext?.time_preset || ''),
    plannerTimeStart: String(plannerContext?.time_range?.start || ''),
    plannerTimeEnd: String(plannerContext?.time_range?.end || ''),
    plannerRisks: Array.isArray(plannerContext?.risks) ? plannerContext.risks.slice().sort() : [],
    weatherRisks: Array.isArray(profile?.weather_risks) ? profile.weather_risks.slice().sort() : [],
    profileHints: Object.keys(profile || {}).sort(),
    requirements,
    monthName,
    timeWindow,
    season: context?.season_hint || '',
    tempBucket: Math.round((context?.weather?.temp || 0) / 2),
    rainBucket: Math.round((context?.weather?.rain_probability || 0) * 10),
    aqiBucket: Math.round((context?.weather?.aqi || 0) / 25),
    rules: ragContext.map((r) => r.id).sort(),
  });
}

function buildPlannerPrompt(context, ragContext) {
  const requirements = String(context?.requirements || 'No additional requirements provided');
  const reasoning = buildReasoningContext(context, ragContext);
  const plannerContext = context?.planner_context || {};
  const userType = String(context?.user_profile?.user_type || 'general').toLowerCase();
  const userToneGuidance = USER_TONE_GUIDANCE[userType] || USER_TONE_GUIDANCE.general;

  const plannerInput = {
    activity: plannerContext?.activity || 'daily_routine',
    date: plannerContext?.date || 'today',
    time_preset: plannerContext?.time_preset || 'custom',
    time_range: plannerContext?.time_range || { start: '09:00', end: '18:00' },
    duration: plannerContext?.duration || null,
    risks: plannerContext?.risks || [],
    notes: plannerContext?.notes || '',
  };

  return [
    'You are an AI Weather Planner. This is an action-planning task, not a generic weather summary.',
    '',
    'Goal:',
    '- Convert Weather + User + Activity + Time into a practical decision plan.',
    '- Focus on feasibility, safest timing, and what user should do next.',
    '',
    'Planner rules (must follow):',
    '- Give a detailed but concise planner answer in 3 lines.',
    '- Hinglish, friendly WhatsApp tone, max 1 emoji per line.',
    '- Mention best time window naturally.',
    '- Mention what to avoid if risk exists.',
    '- Give at least one alternative suggestion if current slot is risky.',
    '- Do not use robotic wording or vague generic lines.',
    '- Do not repeat greeting or raw RAG labels.',
    '',
    'Output structure (plain text, no headings):',
    'Line 1: feasibility verdict for selected activity + current context.',
    'Line 2: clear actionable advice (carry/avoid/use/timing).',
    'Line 3: best time + alternative suggestion if needed.',
    '',
    'User-type nuance:',
    userToneGuidance,
    '',
    'Planner input JSON:',
    JSON.stringify(plannerInput, null, 2),
    '',
    'Reasoning context JSON:',
    JSON.stringify(reasoning, null, 2),
    '',
    'Additional user requirements:',
    requirements,
    '',
    'Output: only final message text.',
  ].join('\n');
}

function fallbackInsight(context, ragContext) {
  if (String(context?.intent || '').toLowerCase() === 'planner') {
    const plannerContext = context?.planner_context || {};
    const activity = String(plannerContext?.activity || 'activity');
    const timeStart = String(plannerContext?.time_range?.start || '09:00');
    const timeEnd = String(plannerContext?.time_range?.end || '18:00');
    const weather = context?.weather || {};
    const temp = Number(weather?.temp || 0);
    const rainProbability = Number(weather?.rain_probability || 0);
    const aqi = Number(weather?.aqi || 0);

    if (rainProbability >= 0.55) {
      return `Hey 👋 ${activity} ke liye is slot me rain risk thoda high lag raha hai 🌧️ Agar zaruri ho to umbrella/raincoat le jaana aur travel buffer rakhna better rahega 👍 Best window possible ho to ${timeEnd} ke baad plan shift karo.`;
    }

    if (temp >= 34 && aqi >= 130) {
      return `Hey 👋 ${activity} abhi possible hai but heat + air quality ka impact reh sakta hai 🌫️ Paani saath rakho, mask use karo aur long outdoor exposure avoid karo 👍 Better option: ${timeStart}-${timeEnd} me shaded/indoor alternative consider karo.`;
    }

    if (temp >= 34) {
      return `Hey 👋 ${activity} ke liye current slot me heat thodi high rahegi ☀️ Direct dhoop avoid karo, hydration maintain rakho aur lightweight gear use karo 👍 Best time: ${timeEnd} ke aas-paas ya shaam ka cooler window.`;
    }

    if (aqi >= 130) {
      return `Hey 👋 ${activity} plan ho sakta hai, but hawa thodi heavy feel ho sakti hai 🌫️ Mask carry karo aur unnecessary outdoor waiting avoid karo 👍 Alternative: short indoor prep karke low-traffic window me niklo.`;
    }

    return `Hey 👋 ${activity} ke liye conditions abhi kaafi manageable lag rahi hain 🙂 Start smooth rakho, essentials saath rakho aur timing disciplined rakhna 👍 Best window: ${timeStart}-${timeEnd}; agar delay ho to next calm slot choose karo.`;
  }

  const weather = context?.weather || {};
  const localHour = Number(context?.local_hour ?? new Date().getHours());
  const timeSlot = getTimeSlotFromHour(localHour);
  const temp = Number(weather?.temp || 0);
  const rainProbability = Number(weather?.rain_probability || 0);
  const aqi = Number(weather?.aqi || 0);

  const isEvening = timeSlot === 'evening' || timeSlot === 'night';
  const hasRainRisk = rainProbability >= 0.5;
  const hasHeatRisk = temp >= 34;
  const hasPollutionRisk = aqi >= 130;
  const hasColdRisk = temp <= 12;

  if (hasRainRisk) {
    return `Hey 👋 ${isEvening ? 'shaam ko weather thoda change ho sakta hai' : 'aaj baarish ka chance bana hua hai'} 🌧️ ${isEvening ? 'Bahar nikalte time umbrella le jaana aur travel me thoda buffer rakhna 👍' : 'Umbrella ya raincoat saath rakhna, commute thoda early plan karna 👍'}`
      .replace(/\s+/g, ' ')
      .trim();
  }

  if (hasHeatRisk && hasPollutionRisk) {
    return 'Hey 👋 aaj dhoop ke saath hawa bhi heavy lag sakti hai 🌫️ Bahar jao toh paani saath rakho, mask use karo aur long outdoor stay avoid karo 👍';
  }

  if (hasHeatRisk) {
    return 'Hey 👋 aaj dhoop kaafi strong hai ☀️ Bahar jao toh paani saath rakho aur direct sun exposure thoda kam rakhna better rahega 👍';
  }

  if (hasPollutionRisk) {
    return 'Hey 👋 aaj hawa thodi heavy feel ho sakti hai 🌫️ Agar bahar zyada time rehna ho toh mask use karo aur hydration maintain rakho 👍';
  }

  if (hasColdRisk) {
    return 'Hey 👋 subah-thand thodi zyada feel ho sakti hai ❄️ Layered kapde pehno aur travel ke liye thoda extra time rakhna safe rahega 👍';
  }

  return `Hey 👋 ${isEvening ? 'shaam ka weather kaafi theek lag raha hai' : 'aaj weather overall balanced lag raha hai'} 🙂 Normal routine continue karo, bas paani saath rakhna aur travel calmly plan karna 👍`
    .replace(/\s+/g, ' ')
    .trim();
}

function buildPersonalizedPrompt(context, ragContext) {
  const requirements = String(context?.requirements || 'No additional requirements provided');
  const reasoning = buildReasoningContext(context, ragContext);
  const userType = String(context?.user_profile?.user_type || 'general').toLowerCase();
  const userToneGuidance = USER_TONE_GUIDANCE[userType] || USER_TONE_GUIDANCE.general;

  return [
    'You are a smart weather assistant who talks like a real person.',
    '',
    'Think like this:',
    '- Understand weather + time + season together',
    '- Decide what matters MOST right now (heat / rain / cold / AQI)',
    '- Then give practical advice',
    '',
    'IMPORTANT:',
    '- Do NOT give generic advice',
    '- Do NOT say "check weather later"',
    '- Do NOT repeat greeting twice',
    '- Do NOT output raw phrases like (Moderate air quality)',
    '',
    'Output Rules:',
    '- 2-3 lines max',
    '- Hinglish, WhatsApp style',
    '- Friendly tone',
    '- Max 1 emoji',
    '- Must give practical actions (umbrella, water, mask, timing)',
    '- Avoid repeating same sentence patterns',
    '',
    'Time + Season Awareness:',
    '- Afternoon -> heat important',
    '- Evening -> travel/rain important',
    '- Winter -> cold/fog',
    '- Monsoon -> rain priority',
    '',
    'User Guidance:',
    userToneGuidance,
    '',
    'Context:',
    JSON.stringify(reasoning, null, 2),
    '',
    'User Needs:',
    requirements,
    '',
    'Output: Only final message',
  ].join('\n');
}

async function callGemini(prompt) {
  const configuredModel = AI_CONFIG.geminiModel || 'gemini-1.5-flash';
  const fallbackModels = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash'];
  let lastSuccessfulModel = null;
  try {
    lastSuccessfulModel = localStorage.getItem(GEMINI_LAST_MODEL_KEY);
  } catch {
    lastSuccessfulModel = null;
  }

  const modelCandidates = [
    lastSuccessfulModel,
    configuredModel,
    ...fallbackModels,
  ].filter((model, index, list) => Boolean(model) && list.indexOf(model) === index);

  let lastError = null;
  for (const modelName of modelCandidates) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${AI_CONFIG.geminiApiKey}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.4, maxOutputTokens: 260 },
      }),
    });

    if (response.ok) {
      const data = await response.json();
      const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || '';
      try {
        localStorage.setItem(GEMINI_LAST_MODEL_KEY, modelName);
      } catch {
        // Ignore localStorage failures.
      }
      return { text, modelName };
    }

    const errorText = await response.text().catch(() => '');
    lastError = new Error(`Gemini request failed for ${modelName}: ${response.status} ${errorText}`);
  }

  throw lastError || new Error('Gemini request failed for all models');
}

export async function generateInsight(context, ragContext) {
  const key = buildCacheKey(context, ragContext);
  const cached = insightCache.get(key);
  if (cached && Date.now() - cached.ts < AI_CONFIG.insightTtlMs) {
    return { text: cached.value, source: 'cache' };
  }

  const isPlannerIntent = String(context?.intent || '').toLowerCase() === 'planner';
  const prompt = isPlannerIntent
    ? buildPlannerPrompt(context, ragContext)
    : buildPersonalizedPrompt(context, ragContext);

  if (AI_CONFIG.llmProvider === 'gemini' && hasGeminiKey()) {
    try {
      const result = await callGemini(prompt);
      const text = sanitizeInsightOutput(result?.text || '');
      const isValid = !looksIncomplete(text) && seemsActionable(text) && !isGeneric(text);
      const value = isValid ? text : fallbackInsight(context, ragContext);
      insightCache.set(key, { ts: Date.now(), value });
      return {
        text: value,
        source: isValid ? `gemini:${result.modelName}` : 'fallback',
      };
    } catch {
      const value = fallbackInsight(context, ragContext);
      insightCache.set(key, { ts: Date.now(), value });
      return { text: value, source: 'fallback' };
    }
  }

  const value = fallbackInsight(context, ragContext);
  insightCache.set(key, { ts: Date.now(), value });
  return { text: value, source: 'fallback' };
}
