const { cache } = require('./cacheService');

const LLM_TTL_MS = 5 * 60 * 1000;

function safeStringify(data) {
  try {
    return JSON.stringify(data, null, 2);
  } catch (_error) {
    return '{}';
  }
}

function buildPrompt({ context, ragContext }) {
  return [
    'You are a weather intelligence assistant.',
    'Generate concise, actionable advice personalized to this user.',
    'Keep response under 80 words and include at least one direct action.',
    '',
    'User and weather context:',
    safeStringify(context),
    '',
    'Retrieved risk knowledge:',
    safeStringify(ragContext),
    '',
    'Output format: plain text only.',
  ].join('\n');
}

function buildCacheKey({ context, ragContext }) {
  const userType = context?.user_profile?.user_type || 'general';
  const weather = context?.weather || {};
  const keyInput = {
    userType,
    weatherMain: weather.weather_main,
    tempBucket: Math.round((weather.temp || 0) / 2),
    rainBucket: Math.round((weather.rain_probability || 0) * 10),
    aqiBucket: Math.round((weather.aqi || 0) / 25),
    ragIds: (ragContext || []).map((r) => r.id).sort(),
  };

  return `llm:${JSON.stringify(keyInput)}`;
}

async function callGemini(prompt) {
  const model = process.env.GEMINI_MODEL || 'gemini-1.5-flash';
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY missing');
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      contents: [{
        parts: [{ text: prompt }],
      }],
      generationConfig: {
        temperature: 0.2,
        maxOutputTokens: 220,
      },
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Gemini error ${response.status}: ${text}`);
  }

  const data = await response.json();
  return data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || '';
}

async function callOpenAI(prompt) {
  const apiKey = process.env.OPENAI_API_KEY;
  const model = process.env.OPENAI_MODEL || 'gpt-4o-mini';

  if (!apiKey) {
    throw new Error('OPENAI_API_KEY missing');
  }

  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      input: [
        {
          role: 'user',
          content: [{ type: 'input_text', text: prompt }],
        },
      ],
      temperature: 0.2,
      max_output_tokens: 220,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenAI error ${response.status}: ${text}`);
  }

  const data = await response.json();
  return data.output_text?.trim() || '';
}

function fallbackAdvice({ context, ragContext }) {
  const weather = context.weather || {};
  const userType = context.user_profile?.user_type || 'general';
  const topRule = ragContext[0]?.text;

  const rainText = weather.rain_probability >= 0.6
    ? 'Rain chances are high.'
    : 'Rain risk is moderate to low.';

  const heatText = weather.temp >= 34
    ? 'Heat can be stressful today.'
    : 'Temperature is manageable.';

  return `${rainText} ${heatText} For ${userType} users, prioritize safe timing, hydration, and route planning. ${topRule || ''}`.trim();
}

async function generateInsight({ context, ragContext }) {
  const cacheKey = buildCacheKey({ context, ragContext });
  const cached = cache.get(cacheKey);
  if (cached) {
    return { text: cached, source: 'cache' };
  }

  const prompt = buildPrompt({ context, ragContext });
  const provider = (process.env.LLM_PROVIDER || 'gemini').toLowerCase();

  let text = '';
  try {
    if (provider === 'openai') {
      text = await callOpenAI(prompt);
    } else {
      text = await callGemini(prompt);
    }
  } catch (error) {
    console.warn('LLM provider failed, using fallback:', error.message);
    text = fallbackAdvice({ context, ragContext });
    cache.set(cacheKey, text, LLM_TTL_MS);
    return { text, source: 'fallback' };
  }

  const finalText = text || fallbackAdvice({ context, ragContext });
  cache.set(cacheKey, finalText, LLM_TTL_MS);
  return { text: finalText, source: provider };
}

module.exports = {
  generateInsight,
};
