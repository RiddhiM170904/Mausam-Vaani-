export const AI_CONFIG = {
  weatherProvider: (import.meta.env.VITE_WEATHER_PROVIDER || 'owm').toLowerCase(),
  openWeatherKey: import.meta.env.VITE_OWM_KEY || '',
  geminiApiKey: import.meta.env.VITE_GEMINI_API_KEY || '',
  geminiModel: import.meta.env.VITE_GEMINI_MODEL || 'gemini-1.5-flash',
  llmProvider: (import.meta.env.VITE_LLM_PROVIDER || 'gemini').toLowerCase(),
  weatherTtlMs: 10 * 60 * 1000,
  insightTtlMs: 5 * 60 * 1000,
};

export const hasWeatherKey = () => Boolean(AI_CONFIG.openWeatherKey);
export const hasGeminiKey = () => Boolean(AI_CONFIG.geminiApiKey);
