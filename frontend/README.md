# Mausam Vaani Frontend

**Modern React weather web app with real-time location-based forecasts**

## ✨ Features

- **Pure dark black theme** with glassmorphism UI
- **Real-time location** auto-detected on every load
- **Guest + Signed-in** user flows
- **10-minute auto-refresh** of weather data
- **Smooth animations** with Framer Motion
- **Interactive weather map** with Leaflet
- **AI insights** for logged-in users
- **Responsive** mobile-first design

## 🚀 Tech Stack

- React 19 + Vite 6
- Tailwind CSS v4
- React Router v7
- Framer Motion
- Recharts (weather graphs)
- Leaflet (interactive maps)
- Axios

## 📦 Quick Start

```bash
# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Add your OpenWeatherMap API key to .env

# Start dev server
npm run dev

# Build for production
npm run build
```

## 🔑 Environment Variables

```env
VITE_OWM_KEY=          # OpenWeatherMap API key (free at openweathermap.org)
VITE_API_URL=          # Backend API URL (default: http://localhost:5000/api)
```

## 📁 Project Structure

```
src/
├── app/              # Router configuration
├── components/       # Reusable UI components
├── context/          # React contexts (Auth, Theme)
├── hooks/            # Custom hooks (useLocation, useWeather)
├── pages/            # Route pages
├── services/         # API clients
└── utils/            # Helpers & formatters
```

## 📱 Pages

| Route | Description |
|-------|-------------|
| `/` | Home dashboard with current weather + AI insights |
| `/forecast` | Hourly/daily forecast with charts |
| `/map` | Interactive weather radar map |
| `/alerts` | Weather warnings & safety tips |
| `/planner` | AI activity planner (signed-in only) |
| `/login` | OTP phone login |
| `/signup` | User onboarding |
| `/profile` | Settings & logout |

## 📍 Real-time Location

The app automatically requests browser geolocation on every load. Falls back to New Delhi if permission denied.

**No manual city search** — always uses live location.

## 🎨 Theme

Pure dark black glassmorphism design inspired by modern weather apps. Minimal, clean, fast.

---

**Built for Mausam Vaani** 🌦️

