// ===============================
// RAG KNOWLEDGE BASE + ENGINE
// ===============================

// -------- 1. KNOWLEDGE BASE --------
export const knowledgeBase = {
  weather_rules: [
    {
      id: 'rain_heavy',
      check: (ctx) => Number(ctx.rain_probability || 0) > 0.6,
      insight: 'Heavy rainfall likely',
      impacts: ['low_visibility', 'road_slippery'],
      severity: 'high',
    },
    {
      id: 'high_temp',
      check: (ctx) => Number(ctx.temperature || 0) > 35,
      insight: 'High temperature conditions',
      impacts: ['heat_stress', 'dehydration'],
      severity: 'medium',
    },
    {
      id: 'low_temp',
      check: (ctx) => Number(ctx.temperature || 0) < 10,
      insight: 'Cold weather conditions',
      impacts: ['cold_stress'],
      severity: 'medium',
    },
    {
      id: 'high_humidity',
      check: (ctx) => Number(ctx.humidity || 0) > 80,
      insight: 'High humidity',
      impacts: ['discomfort'],
      severity: 'low',
    },
    {
      id: 'strong_wind',
      check: (ctx) => Number(ctx.wind_speed || 0) > 20,
      insight: 'Strong winds expected',
      impacts: ['dust', 'instability'],
      severity: 'medium',
    },
  ],

  aqi_rules: [
    {
      id: 'aqi_moderate',
      check: (ctx) => Number(ctx.aqi || 0) >= 100 && Number(ctx.aqi || 0) <= 150,
      insight: 'Moderate air quality',
      advice: ['Sensitive groups reduce outdoor exposure'],
      severity: 'low',
    },
    {
      id: 'aqi_unhealthy',
      check: (ctx) => Number(ctx.aqi || 0) > 150 && Number(ctx.aqi || 0) <= 250,
      insight: 'Unhealthy air quality',
      advice: ['Avoid prolonged outdoor exposure'],
      severity: 'high',
    },
    {
      id: 'aqi_severe',
      check: (ctx) => Number(ctx.aqi || 0) > 250,
      insight: 'Severe air quality',
      advice: ['Stay indoors', 'Use masks'],
      severity: 'high',
    },
  ],

  user_rules: {
    driver: [
      {
        conditions: ['rain_heavy'],
        advice: 'Roads may be slippery, reduce speed',
      },
    ],
    delivery: [
      {
        conditions: ['rain_heavy'],
        advice: 'Expect delivery delays',
      },
    ],
    farmer: [
      {
        conditions: ['rain_heavy'],
        advice: 'Avoid irrigation, rainfall sufficient',
      },
      {
        conditions: ['high_temp'],
        advice: 'Irrigate crops to prevent heat stress',
      },
    ],
    worker: [
      {
        conditions: ['high_temp'],
        advice: 'Avoid working during peak heat hours',
      },
    ],
    office: [
      {
        conditions: ['rain_heavy'],
        advice: 'Plan commute early',
      },
    ],
    student: [
      {
        conditions: ['rain_heavy'],
        advice: 'Carry rain protection',
      },
    ],
    senior: [
      {
        conditions: ['high_temp'],
        advice: 'Avoid afternoon heat exposure',
      },
    ],
    general: [
      {
        conditions: ['high_temp'],
        advice: 'Stay hydrated',
      },
    ],
  },

  combined_rules: [
    {
      conditions: ['high_temp', 'aqi_unhealthy'],
      insight: 'Heat + pollution risk',
      advice: ['Avoid outdoor activity', 'Stay hydrated'],
      severity: 'high',
    },
    {
      conditions: ['rain_heavy', 'strong_wind'],
      insight: 'Storm conditions',
      advice: ['Avoid travel', 'Stay indoors'],
      severity: 'high',
    },
  ],
};

// -------- 2. RULE MATCHING ENGINE --------
function matchRules(context) {
  const triggered = [];

  knowledgeBase.weather_rules.forEach((rule) => {
    if (rule.check(context)) {
      triggered.push(rule.id);
    }
  });

  knowledgeBase.aqi_rules.forEach((rule) => {
    if (rule.check(context)) {
      triggered.push(rule.id);
    }
  });

  return triggered;
}

// -------- 3. RETRIEVE KNOWLEDGE --------
export function retrieveKnowledge(context) {
  const triggeredRules = matchRules(context);

  const insights = [];
  const advice = [];
  const rich = [];

  knowledgeBase.weather_rules.forEach((rule) => {
    if (triggeredRules.includes(rule.id)) {
      insights.push(rule.insight);
      rich.push({ id: rule.id, text: rule.insight, severity: rule.severity || 'medium' });
    }
  });

  knowledgeBase.aqi_rules.forEach((rule) => {
    if (triggeredRules.includes(rule.id)) {
      insights.push(rule.insight);
      rich.push({ id: rule.id, text: rule.insight, severity: rule.severity || 'medium' });
      if (rule.advice) advice.push(...rule.advice);
    }
  });

  knowledgeBase.combined_rules.forEach((rule, index) => {
    const match = rule.conditions.every((conditionId) => triggeredRules.includes(conditionId));
    if (match) {
      insights.push(rule.insight);
      advice.push(...rule.advice);
      rich.push({
        id: `combined_${index + 1}`,
        text: rule.insight,
        severity: rule.severity || 'high',
      });
    }
  });

  const userType = String(context.user_type || 'general').toLowerCase();
  const userRules = knowledgeBase.user_rules[userType] || knowledgeBase.user_rules.general || [];

  userRules.forEach((rule, index) => {
    const match = rule.conditions.every((conditionId) => triggeredRules.includes(conditionId));
    if (match) {
      advice.push(rule.advice);
      rich.push({
        id: `${userType}_advice_${index + 1}`,
        text: rule.advice,
        severity: 'medium',
      });
    }
  });

  return {
    triggeredRules,
    insights,
    advice,
    rich,
  };
}

// -------- 4. BUILD LLM CONTEXT --------
export function buildLLMContext(context) {
  const ragData = retrieveKnowledge(context);

  return {
    weather: {
      temperature: context.temperature,
      rain_probability: context.rain_probability,
      aqi: context.aqi,
      humidity: context.humidity,
      wind_speed: context.wind_speed,
    },
    user: {
      type: context.user_type,
      profile: context.profile || {},
    },
    insights: ragData.insights,
    advice: ragData.advice,
  };
}

// -------- 5. COMPATIBILITY EXPORT --------
// Existing pipeline expects an array of {id, severity, text}.
export function retrieveRagContext(context, limit = 6) {
  const weather = context?.weather || {};
  const user = context?.user_profile || {};

  const normalizedContext = {
    temperature: Number(weather.temp || weather.temperature || 0),
    rain_probability: Number(weather.rain_probability || 0),
    aqi: Number(weather.aqi || 0),
    humidity: Number(weather.humidity || 0),
    wind_speed: Number(weather.wind_speed || 0),
    user_type: String(user.user_type || 'general').toLowerCase(),
    profile: user.profile || {},
  };

  const rag = retrieveKnowledge(normalizedContext);
  const severityRank = { high: 3, medium: 2, low: 1 };

  return rag.rich
    .sort((a, b) => (severityRank[b.severity] || 0) - (severityRank[a.severity] || 0))
    .slice(0, limit);
}
