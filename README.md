# 🌤️Mausam Vaani - AI Weather Prediction & Actionable Advisory Platform (India)

> Making weather predictions **actionable, hyperlocal, inclusive, and personalized** for every Indian citizen—farmers, workers, commuters, policymakers, and businesses. 🇮🇳

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license) 
![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-API-000.svg?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-TimeSeries-FF6F00.svg?logo=tensorflow)
![LLM](https://img.shields.io/badge/Generative_AI-Gemini-blueviolet)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Data-4169E1.svg?logo=postgresql)
![Redis](https://img.shields.io/badge/Redis-Cache-dc382d.svg?logo=redis)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

---

## ✨ Why This Matters

India’s current public weather infrastructure excels at **macro-scale hazard prediction** (cyclones, heatwaves) but falls short for **daily livelihood decisions**:
- District / city-level granularity is too coarse for farms, construction sites, logistics hubs.
- Raw weather metrics push interpretation burden onto users.
- Rural digital divide creates inequitable access.
- Fragmented datasets (IMD + private + community) remain under-fused.
- No unified platform delivering **role-aware, context-driven, multi-lingual recommendations**.

> We bridge the gap from “What is the weather?” to “What should I do now?”

---

## 🚀 Core Value Proposition

| Layer | What It Does | Innovation |
|-------|--------------|------------|
| Hyperlocal Forecasting | Temporal Fusion Transformer (TFT) + nowcasting integration | Street / village-scale micro-climate resolution |
| Data Fusion | IMD feeds + private APIs + low-cost IoT + crowdsourcing | Robust QC & anomaly reconciliation |
| Impact Intelligence | Weather → Activity / Crop / Sector outcomes | Actionability beyond raw metrics |
| Personalization | Gemini LLM + user metadata + predicted weather | On-demand advisory (multi-role, multi-language) |
| Accessibility | Web + SMS + WhatsApp + Voice | Inclusive reach (2G → smartphone) |
| API Ecosystem | Unified JSON advisory responses | Government, enterprise & developer integrations |

---

## 🧠 High-Level Architecture

```
User (Web / SMS / WhatsApp / API)
        │
        ▼
+--------------------+
| Flask API Gateway  |  Auth / Rate limit / Orchestration
+--------------------+
   │        │
   │        ├── User Profile & Preferences (PostgreSQL)
   │
   ├── Weather Prediction Service (TFT DL Model)
   │        │
   │        ├── Data Fusion Layer:
   │        │      • IMD APIs
   │        │      • Private Weather APIs
   │        │      • IoT & Crowdsourced Sensors
   │        │      • External Geo / Crop Datasets
   │        │
   │        └── Redis Cache (hot forecasts)
   │
   └── Personalization Service (Gemini LLM)
             │
             └── Prompt Templates + Guardrails
                   ↓
             Actionable Advisory JSON
```

---


## 🔄 Example Data Flow (End-to-End)

```json
POST /api/get-insight
Request:
{
  "role": "farmer",
  "location": "Lucknow, Uttar Pradesh",
  "coordinates": { "lat": 26.8467, "lon": 80.9462 },
  "crop_type": "rice",
  "activity": "pesticide_spraying",
  "language": "hi"
}

Internal Steps:
1. Retrieve / compute hyperlocal forecast (TFT) → base metrics
2. Inject historical pattern + anomaly flags
3. Build structured prompt for Gemini with role + context
4. LLM returns actionable advice (multi-lingual)
5. Apply safety / compliance post-processing (guardrails)
6. Persist advisory + meta for analytics

Response:
{
  "weather": {
    "temperature": 32.0,
    "humidity": 85,
    "rainfall_probability": 0.9,
    "rain_intensity_mm": 18.4,
    "wind_speed_kmph": 12,
    "condition": "Heavy Rain",
    "confidence": 0.89,
    "time_window": "Next 6 hours"
  },
  "advisory": {
    "short": "आज भारी बारिश होने वाली है, कीटनाशक का छिड़काव रोकें।",
    "action_plan": [
      "छिड़काव 24 घंटे के लिए टालें",
      "जल निकासी नाली साफ करें",
      "परसों सुबह 6–9 बजे नया छिड़काव विंडो"
    ],
    "risk_level": "High",
    "language": "hi"
  },
  "meta": {
    "generated_at": "2025-08-30T10:40:00Z",
    "model_version": "tft_v0.3.1",
    "advice_engine": "gemini_r1_guarded",
    "lat": 26.8467,
    "lon": 80.9462
  }
}
```

---

## 🏗️ Repository Structure

```
root/
├── Frontend/                # ✅ React + Vite + Tailwind UI (COMPLETE)
│   ├── src/
│   │   ├── components/     # Navbar, Footer, FeatureCard, WeatherWidget
│   │   ├── pages/          # Home, Features, About, Contact
│   │   ├── App.jsx         # Router configuration
│   │   └── main.jsx        # Entry point
│   ├── package.json        # Dependencies
│   ├── vite.config.js      # Vite configuration
│   ├── tailwind.config.js  # Tailwind setup
│   └── README.md           # Frontend documentation
├── Backend/                 # Flask API (auth, routing, orchestration) - TO BE CREATED
├── AI-Backend/              # DL models + LLM prompt layer - TO BE CREATED
├── docs/                    # Technical & user documentation
├── deployment/              # Docker, K8s, infra scripts
├── monitoring/              # Prometheus, Grafana configs
└── scripts/                 # Utilities (ingestion, maintenance)
```

### ✅ Frontend Status: **COMPLETE**
- **Development server running** at `http://localhost:3000/`
- **All core pages** implemented with responsive design
- **Navigation** with mobile menu
- **329 packages** installed, 0 vulnerabilities
- **Ready for backend integration**

See `Frontend/SETUP_COMPLETE.md` for detailed documentation.

---

## 🧩 Key Features

### Forecast Engine
- Temporal Fusion Transformer (multi-horizon)
- Optional nowcasting via ConvLSTM / optical flow (future)
- Spatio-temporal feature engineering (lat/lon embedding + cyclical time features)
- Confidence scoring + anomaly tagging

### Personalization Layer
- Gemini LLM with occupation templates
- Safety guardrails (e.g., no dangerous chemical misuse)
- Multi-lingual expansions (12+ Indian languages roadmap)
- Role verticals: Farming, Construction, Logistics, Outdoor Events, Public Health

### Data Quality & Validation
| Stage | Method |
|-------|--------|
| Sensor Integrity | Drift detection, z-score filtering |
| Cross-Source Reconciliation | Median blending + outlier rejection |
| Spatial Consistency | Kriging / IDW fallback |
| Confidence Metric | Ensemble dispersion + historical baseline deviation |

### Accessibility
- SMS (compressed advisory form)
- WhatsApp conversational agent
- Voice (TTS for low literacy contexts)
- Offline cache (PWA manifest + localStorage)

---

## 💽 Data Inputs (Illustrative)

| Category | Examples |
|----------|----------|
| Core Meteorology | Temperature, humidity, pressure, rainfall, wind |
| Derived Indices | Heat index, dew point, evapotranspiration proxy |
| Geo Features | Elevation, land use type, distance to water |
| Temporal | Hour of day, day-of-year, monsoon phase, holiday flag |
| Crop Context (agri vertical) | Crop type, growth stage, soil moisture (future) |

---

## 🔐 Security & Governance

| Aspect | Approach |
|--------|----------|
| Auth | JWT (short-lived) + refresh tokens |
| Rate Limiting | IP + user-tier (Redis) |
| PII Handling | Minimal storage; hashed identifiers |
| LLM Safety | Prompt guard + regex redaction + post-filter |
| Compliance (Future) | MeitY guidelines, data residency |

---

## 📦 Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | React, Tailwind CSS, Axios, i18next |
| API | Flask, Flask-JWT-Extended, Marshmallow |
| ML/DL | TensorFlow / PyTorch (TFT), Pandas, NumPy |
| LLM | Gemini API (structured prompting) |
| Storage | PostgreSQL (users), MongoDB (raw weather), Redis (cache), Object Storage (artifacts) |
| Messaging (future) | NATS / Kafka for streaming ingestion |
| Observability | Prometheus, Grafana, Structured JSON logs |
| CI/CD (future) | GitHub Actions → Docker Registry → K8s |

---

## ⚙️ Getting Started

### Frontend (✅ Ready Now!)

The frontend is complete and running. To view it:

```bash
cd Frontend
npm install  # Already done if following setup
npm run dev  # Development server at http://localhost:3000
```

**What you get:**
- Modern, responsive React application
- 4 complete pages: Home, Features, About, Contact
- Professional UI with Tailwind CSS
- Mobile-friendly navigation
- Ready for backend API integration

See `Frontend/SETUP_COMPLETE.md` for full details.

### Backend & AI Services (Coming Soon)

```bash
# Backend API (Planned)
cd Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py

# AI/ML Service (Planned)
cd AI-Backend
pip install -r requirements.txt
```

---

## 🔑 Environment Variables (Core Subset)

| Variable | Description |
|----------|-------------|
| GEMINI_API_KEY | LLM access key |
| DATABASE_URL | PostgreSQL connection |
| REDIS_URL | Cache endpoint |
| JWT_SECRET | Token signing secret |
| ALLOWED_ORIGINS | CORS domains |
| ENABLE_LLM | Toggle personalization module |
| WEATHER_FUSION_MODE | 'hybrid' / 'imd_only' / 'cache' |

(See `.env.example` in each service directory)

---

## 🧪 Testing Strategy

| Layer | Tests |
|-------|-------|
| API | Pytest: contract, auth, rate limit |
| ML | Forecast MAE/MAPE validation, drift checks |
| LLM | Advisory structure + toxicity filter tests |
| E2E | Simulated user journeys (CLI / Postman collection) |

---

## 📊 Metrics & Observability (Future)

| Metric | Purpose |
|--------|---------|
| Forecast Error Trend | Model retrain triggers |
| Advisory Acceptance Rate | Personalization quality |
| Latency (P50/P95) | User experience |
| Sensor Dropout Rate | Data integrity alerts |
| LLM Token Cost / User | Cost optimization |

---

## 🧬 Model Notes (TFT)

| Aspect | Current |
|--------|---------|
| Horizon | 6–24 hours (configurable) |
| Resolution | Up to sub-km (with grid interpolation) |
| Retrain Cadence | Weekly (rolling window) |
| Loss | Quantile (P10/P50/P90) |
| Feature Importance | Attention interpretability logging |

---

## 🎯 Roadmap (High-Level)

| Phase | Focus |
|-------|-------|
| Phase 1 | Core forecast + basic advisory (Farmer vertical) |
| Phase 2 | Multi-role templates + SMS + caching |
| Phase 3 | Government & enterprise dashboards |
| Phase 4 | Sensor network pilot + nowcasting |
| Phase 5 | API monetization + multilingual scaling |

Detailed: `docs/roadmap.md` (planned)

---

## 🤖 Example Prompt (LLM Personalization)

```
System: You are a safety-aware Indian agro-meteorological assistant.
User Context:
- Role: Farmer
- Crop: Rice (tillering)
- Location: 26.8467, 80.9462 (Lucknow)
Weather (Next 6h):
- Temp: 32°C
- Rain probability: 90% (heavy)
- Humidity: 85%
Task: Provide concise actionable advice in Hindi. Avoid recommending pesticide spraying in impending rain. Include 1 risk warning.
Format: JSON with keys: weather_summary, advice, risk_alert
```

---

## 🧭 API (Preview)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/auth/register` | POST | Create user |
| `/api/auth/login` | POST | Obtain JWT |
| `/api/get-insight` | POST | Weather + advisory bundle |
| `/api/forecast/raw` | POST | Raw forecast (dev use) |
| `/api/user/preferences` | GET/PUT | Role / language / sector |
| `/api/health` | GET | Liveness |

Example:
```bash
curl -X POST https://api.example.com/api/get-insight \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"role":"farmer","coordinates":{"lat":26.84,"lon":80.94},"crop_type":"rice"}'
```

---

## 🤝 Contributing

We welcome:
- Data pipeline improvements
- Regional language advisory templates
- Model optimization (quantization / pruning)
- Sensor QA heuristics

See: `docs/development/contributing.md` (planned)

---

## 👥 Contributors

| GitHub Handle | Name | Role |
|---------------|------|------|
| [@5uhani](https://github.com/5uhani) | Suhani | Collaborator |
| [@aka-aksh](https://github.com/aka-aksh) | Akshjeet Gakrey | Contributor |
| [@Mradul-code](https://github.com/Mradul-code) | Mradul-code | Collaborator |
| [@RiddhiM170904](https://github.com/RiddhiM170904) | Riddhi Mhadgut | Collaborator |
| [@shreyad2806](https://github.com/shreyad2806) | Shreya Dubey | Collaborator |

> Want to join? Open a discussion or issue with the tag `onboarding`.

---

## 📝 License

Licensed under the **MIT License**. You are free to use, modify, distribute with attribution. See `LICENSE`.

---

## 🙏 Acknowledgements

- IMD public datasets & open meteorological research community
- Open-source time-series forecasting pioneers
- Farmers, logistics operators, and civic workers informing real-world use cases
- Contributors exploring inclusive AI for Bharat

---

## 📬 Contact

- Issues: GitHub Issues tab  
- Partnerships: dhairyag31@gmail.com 
- Advisory Feedback: dhairyag31@gmail.com 

---

> “Accurate information is power only when translated into timely action—our platform closes that loop.”

🌱 Built for resilience. Designed for impact.