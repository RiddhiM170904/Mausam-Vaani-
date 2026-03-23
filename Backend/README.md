# Mausam Vaani Backend

Node + Express backend for weather intelligence with a production-style MVP pipeline:

1. User profile and location
2. Current weather APIs (OpenWeather)
3. RAG knowledge retrieval (JSON rules)
4. Context builder
5. LLM reasoning (Gemini/OpenAI with fallback)
6. Event-based notification triggers

## New Endpoint: Personalized Intelligence

### POST /api/intelligence/insight

Builds end-to-end personalized advice.

Request body (if not authenticated):

```json
{
	"location": { "lat": 23.25, "lon": 77.41 },
	"userProfile": {
		"user_id": "U1",
		"user_type": "driver",
		"profile": {
			"vehicle": "bike",
			"distance": 10
		}
	}
}
```

Response:

```json
{
	"success": true,
	"data": {
		"insight": "Heavy rain expected. Since you commute by bike, carry rain protection and avoid non-urgent travel in peak rainfall windows.",
		"llmSource": "gemini",
		"weather": {
			"temp": 36,
			"rain_probability": 0.7,
			"aqi": 180,
			"wind_speed": 12,
			"humidity": 70
		},
		"context": {},
		"ragContext": [],
		"notifications": []
	}
}
```

### POST /api/intelligence/notifications/preview

Evaluates trigger rules without calling the LLM.

## Merged Notification Backend

Notification-Backend features are now merged into this same `Backend` service.

### Notification APIs

1. `POST /api/subscriptions`
2. `DELETE /api/subscriptions`
3. `POST /api/preferences`
4. `POST /api/notifications/test`
5. `POST /api/notifications/run-now`
6. `GET /api/notifications/status`

### Scheduler

- In-process scheduler runs every 5 minutes (configurable via `NOTIFICATION_CRON`).
- Scheduler auto-starts on server boot only when notification env vars are configured.
- If notification env vars are missing, main backend still runs and notification APIs return `503`.

## Environment Variables

Create a `.env` in `Backend`:

```env
NODE_ENV=development
PORT=5000
MONGODB_URI=mongodb://127.0.0.1:27017/mausam_vaani
JWT_SECRET=replace_me
JWT_EXPIRE=30d
FRONTEND_URL=http://localhost:5173

OPENWEATHER_API_KEY=replace_me

# Choose one provider: gemini or openai
LLM_PROVIDER=gemini

# Gemini
GEMINI_API_KEY=replace_me
GEMINI_MODEL=gemini-1.5-flash

# OpenAI
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Notification service (Supabase + Web Push)
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
VAPID_SUBJECT=mailto:you@example.com
VAPID_PUBLIC_KEY=
VAPID_PRIVATE_KEY=
NOTIFICATION_TIMEZONE=Asia/Kolkata
NOTIFICATION_CRON=*/5 * * * *
```

## Run

```bash
npm install
npm run dev
```

## Key Files

- `src/routes/intelligence.js`: API orchestration for context, RAG, LLM, and notifications
- `src/services/weatherService.js`: Real-time weather + AQI + caching
- `src/services/contextService.js`: Structured reasoning context builder
- `src/services/ragService.js`: Rule retrieval for MVP RAG
- `src/services/llmService.js`: LLM provider integration + fallback
- `src/services/notificationService.js`: Event trigger engine
- `src/data/rag_rules.json`: Knowledge rules

### Merged Notification Files

- `src/notifications/*`: scheduler, push, subscription services and config
- `src/routes/notificationSubscriptions.js`: subscription + preference APIs
- `src/routes/notificationJobs.js`: test push + run-now APIs
- `sql/notification_schema.sql`: required Supabase notification tables
