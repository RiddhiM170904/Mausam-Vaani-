# 🚀 Mausam-Vaani API - Quick Start Guide

## 📋 Prerequisites

- Python 3.8+
- Trained model file (`best_model.pth`)
- Gemini API key

---

## ⚡ Quick Setup (3 Steps)

### 1. Install Dependencies

```bash
cd AI-Backend
pip install -r requirements_api.txt
```

### 2. Configure Environment

Create `.env` file:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Gemini API key
```

**Required in `.env`:**
```env
GEMINI_API_KEY=your_actual_api_key_here
MODEL_PATH=best_model.pth
```

### 3. Run the Server

```bash
python app.py
```

Or with uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Server will start at:** `http://localhost:8000`

**API Docs:** `http://localhost:8000/docs`

---

## 🔑 Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key
4. Add to `.env` file:
   ```env
   GEMINI_API_KEY=AIza...your_key_here
   ```

---

## 📡 API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Weather Prediction + Insights
```bash
POST http://localhost:8000/predict
```

**Request Body:**
```json
{
  "weather_input": {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "location_name": "Delhi"
  },
  "user_context": {
    "profession": "Farmer",
    "additional_context": {
      "crop": "Rice"
    }
  },
  "forecast_hours": 24
}
```

**Response:**
```json
{
  "location": "Delhi",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "current_time": "2024-11-24T16:00:00",
  "forecast_hours": 24,
  "predictions": [
    {
      "timestamp": "2024-11-24T17:00:00",
      "temperature": 28.5,
      "humidity": 65.2,
      "wind_speed": 5.3,
      "rainfall": 0.0,
      "pressure": 1010.2,
      "cloud_cover": 45.0
    }
    // ... 23 more hourly predictions
  ],
  "summary": {
    "avg_temperature": 27.8,
    "min_temperature": 24.5,
    "max_temperature": 32.1,
    "avg_rainfall": 2.3,
    "total_rainfall": 55.2,
    "avg_humidity": 68.5,
    "avg_wind_speed": 5.1
  },
  "personalized_insight": "🌧️ Heavy rainfall expected in the next 24 hours. Avoid pesticide spraying and postpone harvesting. Ensure proper drainage in your rice fields to prevent waterlogging.",
  "profession": "Farmer"
}
```

---

## 🎯 Supported Professions

The API provides personalized insights for:

- **Farmer** - Crop-specific advice, irrigation, pest control
- **Commuter** - Traffic warnings, travel times, safety
- **Construction Worker** - Work schedule, safety warnings
- **Outdoor Sports** - Activity timing, hydration, safety
- **General** - General weather advice for everyone

---

## 🧪 Test the API

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "weather_input": {
      "latitude": 28.6139,
      "longitude": 77.2090,
      "location_name": "Delhi"
    },
    "user_context": {
      "profession": "Farmer",
      "additional_context": {"crop": "Rice"}
    },
    "forecast_hours": 24
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "weather_input": {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "location_name": "Delhi"
    },
    "user_context": {
        "profession": "Farmer",
        "additional_context": {"crop": "Rice"}
    },
    "forecast_hours": 24
}

response = requests.post(url, json=data)
print(response.json())
```

### Using JavaScript (Frontend)

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    weather_input: {
      latitude: 28.6139,
      longitude: 77.2090,
      location_name: 'Delhi'
    },
    user_context: {
      profession: 'Farmer',
      additional_context: { crop: 'Rice' }
    },
    forecast_hours: 24
  })
});

const data = await response.json();
console.log(data);
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Your Gemini API key |
| `MODEL_PATH` | No | `best_model.pth` | Path to trained model |
| `HOST` | No | `0.0.0.0` | Server host |
| `PORT` | No | `8000` | Server port |
| `ALLOWED_ORIGINS` | No | `*` | CORS allowed origins |

### Model Configuration

The model configuration is hardcoded in `app.py`:

```python
MODEL_CONFIG = {
    'num_features': 9,
    'hidden_dim': 128,
    'num_heads': 4,
    'num_layers': 2,
    'forecast_horizon': 24,
    'output_dim': 6,
    'dropout': 0.1,
}
```

**⚠️ Important:** These must match your training configuration!

---

## 📊 API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation where you can:
- View all endpoints
- See request/response schemas
- Test API calls directly

---

## 🐛 Troubleshooting

### Issue: "Model not loaded"

**Solution:** Check that `best_model.pth` exists in the AI-Backend directory

```bash
ls -lh best_model.pth
```

### Issue: "Gemini API key not configured"

**Solution:** Add your API key to `.env` file

```env
GEMINI_API_KEY=your_actual_key_here
```

### Issue: "Module not found"

**Solution:** Install all dependencies

```bash
pip install -r requirements_api.txt
```

### Issue: "CUDA out of memory"

**Solution:** The model will automatically use CPU if CUDA is not available. This is normal.

### Issue: "Port 8000 already in use"

**Solution:** Use a different port

```bash
uvicorn app:app --port 8001
```

---

## 🚀 Production Deployment

### Using Gunicorn (Recommended)

```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY app.py .
COPY best_model.pth .
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t mausam-vaani-api .
docker run -p 8000:8000 mausam-vaani-api
```

---

## 📝 Notes

### Historical Data

The API currently uses **dummy historical data** for demo purposes. In production, you should:

1. Integrate with a weather data provider (OpenWeather, Weather API, etc.)
2. Store historical data in a database
3. Pass real historical data in the `historical_data` field

### CORS

The API allows all origins (`*`) by default. In production:

1. Update `ALLOWED_ORIGINS` in `.env`:
   ```env
   ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
   ```

2. Or modify `app.py`:
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements_api.txt`)
- [ ] `.env` file created with `GEMINI_API_KEY`
- [ ] `best_model.pth` file present
- [ ] Server starts without errors
- [ ] API docs accessible at `/docs`
- [ ] Test prediction successful
- [ ] Frontend can connect to API

---

## 🎉 You're Ready!

Your FastAPI server is now running with:
- ✅ TFT weather prediction model
- ✅ Gemini LLM personalized insights
- ✅ CORS enabled for frontend
- ✅ Interactive API documentation
- ✅ Production-ready error handling

**Next steps:**
1. Connect your React frontend to `http://localhost:8000`
2. Test with different professions and locations
3. Deploy to production when ready!

---

**Need help?** Check the API docs at http://localhost:8000/docs
