# Frontend AI Modules

This folder is now frontend-only JavaScript logic. No separate backend deployment and no extra installation are required.

## What it does

- Fetches weather + AQI directly from OpenWeather using `VITE_OWM_KEY`
- Builds context from user profile + weather
- Retrieves local RAG rules
- Generates personalized insight (Gemini optional via `VITE_GEMINI_API_KEY`)
- Builds notification triggers

## Files

- `index.js`: Public exports
- `aiPipeline.js`: Main pipeline functions
- `weatherApi.js`: OpenWeather calls + weather cache
- `contextBuilder.js`: User/context normalization
- `ragRules.js`: Built-in rule base
- `ragEngine.js`: Rule retrieval
- `llmClient.js`: Gemini/fallback insight generation
- `notificationEngine.js`: Trigger logic
- `aiConfig.js`: Reads Vite env values

## Usage in frontend code

```js
import { getPersonalizedInsight, previewNotifications } from '../AI/index.js';

const insightResponse = await getPersonalizedInsight({
  user_id: 'your-supabase-user-id',
  location: { lat: 23.25, lon: 77.41 },
  user_profile: {
    user_id: 'U1',
    user_type: 'driver',
    profile: { vehicle: 'bike', distance: 10 }
  },
  requirements: 'Keep advice short, actionable, and prioritize travel safety.'
});

const notificationResponse = previewNotifications({
  weather: {
    temp: 36,
    rain_probability: 0.7,
    aqi: 180,
    wind_speed: 12
  },
  user_profile: { user_type: 'driver' }
});
```

## Required frontend env

- `VITE_OWM_KEY`

## Optional frontend env

- `VITE_GEMINI_API_KEY`
- `VITE_GEMINI_MODEL`
- `VITE_LLM_PROVIDER`

If `user_id` is provided in payload, AI modules fetch user details from Supabase `users` table and merge with payload user details before LLM call.
