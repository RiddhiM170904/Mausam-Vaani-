# 🚀 Complete Setup & Run Guide - Mausam Vaani

## Quick Start Commands

### Option 1: Run Everything (Recommended for Demo)

```powershell
# Terminal 1: Start Frontend
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\Frontend"
npm run dev

# Terminal 2: Start AI Backend
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\AI-Backend"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# Copy .env.example to .env and add GEMINI_API_KEY
uvicorn app:app --reload --port 8000
```

---

## 📋 Detailed Step-by-Step Setup

### 1️⃣ Frontend Setup (Already Done! ✅)

The Frontend is **complete and running** at http://localhost:3000

**If you need to restart:**
```powershell
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\Frontend"
npm run dev
```

**Files Created:**
- ✅ All pages: Home, Features, About, Contact, **Demo (NEW!)**
- ✅ API integration: `src/services/weatherApi.js`
- ✅ API config: `src/config/api.js`
- ✅ Environment: `.env` with `VITE_API_URL=http://localhost:8000`

---

### 2️⃣ AI Backend Setup (FastAPI)

The AI Backend has TWO app files:
- `app.py` (root) - **FastAPI** - ✅ **USE THIS ONE**
- `api/app.py` - Flask - Alternative

**Step 1: Navigate to AI-Backend**
```powershell
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\AI-Backend"
```

**Step 2: Create Virtual Environment**
```powershell
python -m venv venv
```

**Step 3: Activate Virtual Environment**
```powershell
.\venv\Scripts\activate
```

**Step 4: Install Dependencies**
```powershell
pip install -r requirements.txt
```

If you get errors, install individually:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-dotenv
pip install google-generativeai
pip install numpy pandas scikit-learn
```

**Step 5: Create .env File**
```powershell
copy .env.example .env
```

**Step 6: Edit .env and Add Your Gemini API Key**

Open `.env` file and add:
```
GEMINI_API_KEY=your_actual_api_key_here
MODEL_PATH=best_model.pth
PORT=8000
```

**Get Gemini API Key:**
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Copy and paste into `.env`

**Step 7: Run the FastAPI Server**
```powershell
uvicorn app:app --reload --port 8000
```

**Alternative: Using Python directly**
```powershell
python app.py
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
Model Path: best_model.pth
Device: cpu
Gemini API: Configured
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

---

### 3️⃣ Test the Integration

**Step 1: Open Frontend**
```
http://localhost:3000/demo
```

**Step 2: Fill the Demo Form:**
- Location: Delhi (or any city)
- Latitude: 28.6139
- Longitude: 77.2090
- Profession: Farmer
- Crop: Rice
- Forecast Hours: 24

**Step 3: Click "Get Weather Prediction"**

You should see:
- ✅ API Status: "API Connected" (green dot)
- ✅ Weather predictions for next N hours
- ✅ Personalized advisory from Gemini AI
- ✅ Weather summary (temperature, rainfall, humidity, wind)

---

## 🔧 Troubleshooting

### Frontend Issues

**Port 3000 already in use:**
```powershell
# Kill the process
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process

# Or use a different port
npm run dev -- --port 3001
```

**API not connecting:**
- Check if backend is running at http://localhost:8000
- Verify `.env` file has `VITE_API_URL=http://localhost:8000`
- Restart frontend after changing `.env`

### Backend Issues

**Port 8000 already in use:**
```powershell
# Use a different port
uvicorn app:app --reload --port 8001

# Update Frontend .env
VITE_API_URL=http://localhost:8001
```

**Gemini API Error:**
```
Error: Gemini API key not configured
```
**Solution:**
- Verify `GEMINI_API_KEY` in `.env` file
- Check API key is valid at https://aistudio.google.com/app/apikey
- Restart backend after adding key

**Model not found:**
```
Error: Model not loaded
```
**Solution:**
- The app works WITHOUT a trained model (uses dummy data)
- For real predictions, train model first:
  ```powershell
  python models/train.py --config config/model_config.yaml
  ```

**Python version issue:**
```powershell
# Check Python version (need 3.10+)
python --version

# If using older Python, upgrade or use conda
```

**Torch installation error:**
```powershell
# Install CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CORS Errors

If you see CORS errors in browser console:

**Backend (app.py) already has CORS enabled:**
```python
allow_origins=["*"]  # Allows all origins
```

If still issues, verify backend console shows:
```
INFO:     127.0.0.1:xxxxx - "POST /predict HTTP/1.1" 200 OK
```

---

## 📊 Project Status

| Component | Status | URL | Port |
|-----------|--------|-----|------|
| Frontend | ✅ Running | http://localhost:3000 | 3000 |
| AI Backend (FastAPI) | ⚙️ Setup Required | http://localhost:8000 | 8000 |
| Backend (Flask) | 📝 Empty (Future) | - | - |

---

## 🎯 API Endpoints

### Backend API (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Weather prediction + insights |
| `/docs` | GET | Swagger API documentation |

### Example API Request

```json
POST http://localhost:8000/predict
Content-Type: application/json

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

### Example API Response

```json
{
  "location": "Delhi",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "current_time": "2025-11-25T10:30:00",
  "forecast_hours": 24,
  "predictions": [
    {
      "timestamp": "2025-11-25T11:00:00",
      "temperature": 25.3,
      "humidity": 66.0,
      "wind_speed": 5.1,
      "rainfall": 0.0,
      "pressure": 1009.0,
      "cloud_cover": 22.0
    }
  ],
  "summary": {
    "avg_temperature": 25.5,
    "min_temperature": 23.2,
    "max_temperature": 28.1,
    "avg_rainfall": 2.3,
    "total_rainfall": 55.2,
    "avg_humidity": 65.4,
    "avg_wind_speed": 5.3
  },
  "personalized_insight": "🌧️ Heavy rainfall expected in the next 24 hours. Avoid pesticide spraying and ensure proper drainage in rice fields. Good time to prepare for sowing once rain subsides.",
  "profession": "Farmer"
}
```

---

## 🌐 URLs Summary

### Development URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend Home | http://localhost:3000/ | Landing page |
| Frontend Demo | http://localhost:3000/demo | Live prediction demo |
| Backend API | http://localhost:8000/ | API root |
| API Docs | http://localhost:8000/docs | Interactive API docs |
| Health Check | http://localhost:8000/health | API status |

---

## 📝 Complete Command Sequence

**Run these in ORDER:**

### Terminal 1: Frontend (Already Running)
```powershell
# If not running, start with:
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\Frontend"
npm run dev
```

### Terminal 2: AI Backend
```powershell
# Navigate
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\AI-Backend"

# Create & activate virtual environment (first time only)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Create .env file (first time only)
copy .env.example .env
# Edit .env and add GEMINI_API_KEY

# Start server
uvicorn app:app --reload --port 8000
```

### Terminal 3: Testing (Optional)
```powershell
# Test health endpoint
curl http://localhost:8000/health

# Or open in browser
start http://localhost:8000/docs
```

---

## ✅ Success Checklist

- [ ] Frontend running at http://localhost:3000
- [ ] Backend running at http://localhost:8000
- [ ] Backend health check shows "healthy"
- [ ] Gemini API key configured in `.env`
- [ ] Demo page shows "API Connected" status
- [ ] Can submit prediction request
- [ ] Receive weather predictions
- [ ] Receive personalized insights

---

## 🎉 You're Ready!

1. ✅ Frontend: http://localhost:3000
2. ✅ Demo Page: http://localhost:3000/demo
3. ✅ API Docs: http://localhost:8000/docs

**Try the demo:**
- Go to http://localhost:3000/demo
- Enter location details
- Select profession
- Click "Get Weather Prediction"
- See AI-powered personalized weather insights!

---

## 📚 Next Steps

1. **Get real weather data** for training:
   - See `AI-Backend/data/data_collection_guide.md`
   - Use Visual Crossing or OpenWeatherMap API

2. **Train the model:**
   ```powershell
   cd AI-Backend
   python models/train.py --config config/model_config.yaml
   ```

3. **Deploy:**
   - Frontend: Vercel, Netlify
   - Backend: Railway, Render, AWS

---

**Need Help?**
- Frontend docs: `Frontend/DOCUMENTATION.md`
- Backend docs: `AI-Backend/README.md`
- API docs: http://localhost:8000/docs
