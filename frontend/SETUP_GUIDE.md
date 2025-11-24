# Frontend-Backend Integration Guide

## 🎉 Integration Complete!

The Mausam-Vaani frontend is now fully integrated with the AI backend. You can get real-time weather predictions and personalized insights.

## 📋 Commands to Run

### Terminal 1: Start AI Backend

```bash
# Navigate to AI-Backend
cd "c:\personal dg\github_repo\Mausam-Vaani-\AI-Backend"

# Activate virtual environment (if not already activated)
.\venv\Scripts\activate

# Make sure .env file exists with GEMINI_API_KEY
# Create .env from .env.example if needed
copy config\.env.example .env
# Then edit .env and add your Gemini API key

# Start Flask API
python api/app.py

# Backend will run on http://localhost:5000
```

### Terminal 2: Start Frontend

```bash
# Navigate to frontend
cd "c:\personal dg\github_repo\Mausam-Vaani-\frontend"

# Install axios dependency
npm install axios

# Start development server
npm run dev

# Frontend will run on http://localhost:5173
```

## 🌐 Usage

1. **Open browser**: Navigate to `http://localhost:5173/dashboard`

2. **Select options**:
   - Choose a city (Delhi, Mumbai, or Bengaluru)
   - Select your profession (Farmer, Commuter, Construction Worker, etc.)
   - Enter context (crop type, transport mode, etc.)

3. **Get insights**: Click "Get Weather Insights" button

4. **View results**:
   - See personalized AI-generated advice from Gemini
   - View 24-hour weather forecast with hourly predictions
   - Check current weather conditions

## ✅ Features Implemented

### 1. API Configuration (`src/config/api.config.js`)
- Backend URL configuration
- API endpoint definitions
- Timeout and retry settings

### 2. API Service Layer (`src/services/api.js`)
- `checkHealth()` - Check backend status
- `predictWeather()` - Get weather predictions
- `getInsight()` - Get personalized insights
- Axios interceptors for logging and error handling

### 3. Mock Weather Data (`src/utils/mockWeatherData.js`)
- Generates realistic 168 hours of historical data
- Supports Delhi, Mumbai, Bengaluru
- Realistic patterns for temperature, humidity, rainfall

### 4. Weather Dashboard (`src/pages/WeatherDashboard.jsx`)
- City selection dropdown
- Profession selection
- Dynamic context input
- Backend status indicator
- Loading states
- Error handling
- Results display

### 5. Weather Forecast Component (`src/components/WeatherForecast.jsx`)
- 24-hour hourly predictions
- Weather icons based on conditions
- Detailed metrics per hour
- Summary statistics

### 6. Insight Card Component (`src/components/InsightCard.jsx`)
- Personalized AI advice display
- Current weather stats
- Location information
- Profession-specific formatting

## 🔧 How It Works

```
User selects city & profession
         ↓
Dashboard generates 168 hours of mock historical data
         ↓
Calls /api/get-insight with:
  - Location (lat, lon, city)
  - User profession
  - User context
  - Historical weather data
         ↓
Backend processes request:
  - DL Model predicts next 24 hours (or uses mock if model not trained)
  - Gemini AI generates personalized advice
         ↓
Frontend displays:
  - 24-hour forecast
  - Personalized insights
  - Current conditions
```

## 🎯 Next Steps

### Immediate
1. ✅ Install backend dependencies (`pip install -r requirements.txt`)
2. ✅ Set up Gemini API key in `.env`
3. ✅ Start backend server
4. ✅ Install frontend dependencies (`npm install axios`)
5. ✅ Start frontend server
6. ✅ Test the dashboard

### Future Enhancements
- Train the DL model with real data
- Add more cities
- Implement user authentication
- Save user preferences
- Add charts and visualizations
- Export forecast data
- Add notifications/alerts
- Mobile app version

## 🐛 Troubleshooting

### Backend not connecting
- Check if Flask server is running on port 5000
- Verify CORS is enabled in `api/app.py`
- Check browser console for errors

### No insights generated
- Verify Gemini API key is set in `.env`
- Check backend logs for errors
- Ensure API key has proper permissions

### Forecast not showing
- Check if API response includes `forecast` array
- Verify data format matches expected structure
- Check browser console for JavaScript errors

## 📊 API Response Format

### Insight Response:
```json
{
  "success": true,
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "city": "Delhi"
  },
  "weather": {
    "current": {
      "temperature": 28.5,
      "humidity": 65,
      "wind_speed": 5.2,
      "rainfall": 0.0,
      "pressure": 1005,
      "cloud_cover": 20
    },
    "condition": "Clear"
  },
  "personalized_insight": "AI-generated advice here...",
  "forecast": [
    {
      "timestamp": "2024-11-24T15:00:00",
      "hour_ahead": 1,
      "temperature": 28.3,
      "humidity": 66,
      ...
    }
  ]
}
```

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Live backend status indicator
- **Loading States**: Visual feedback during API calls
- **Error Messages**: Clear error descriptions
- **Professional UI**: Modern gradient backgrounds and card designs
- **Accessible**: Proper labels and semantic HTML

---

**Ready to use!** Just start both servers and navigate to the dashboard. 🚀
