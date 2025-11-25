# ⚡ Quick Start - Mausam Vaani

## 🚀 Run Commands

### Start Frontend (Terminal 1)
```powershell
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\Frontend"
npm run dev
```
✅ **Already running at** http://localhost:3000

### Start Backend (Terminal 2)
```powershell
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\AI-Backend"
.\venv\Scripts\activate
uvicorn app:app --reload --port 8000
```
📍 **Will run at** http://localhost:8000

---

## 🔧 First Time Setup (Backend Only)

### 1. Create Virtual Environment
```powershell
cd "c:\Riddhi\Github Repo\Mausam-Vaani-\AI-Backend"
python -m venv venv
```

### 2. Activate Environment
```powershell
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment
```powershell
copy .env.example .env
```

Edit `.env` and add:
```
GEMINI_API_KEY=your_key_here
```

Get key from: https://aistudio.google.com/app/apikey

### 5. Run Server
```powershell
uvicorn app:app --reload --port 8000
```

---

## 🌐 URLs

| Service | URL | Status |
|---------|-----|--------|
| Frontend | http://localhost:3000 | ✅ Running |
| Demo Page | http://localhost:3000/demo | ✅ New! |
| Backend API | http://localhost:8000 | ⚙️ Setup needed |
| API Docs | http://localhost:8000/docs | ⚙️ Setup needed |
| Health | http://localhost:8000/health | ⚙️ Setup needed |

---

## ✅ Test It!

1. Open http://localhost:3000/demo
2. Enter location: **Delhi** (28.6139, 77.2090)
3. Select profession: **Farmer**
4. Click "Get Weather Prediction"
5. See AI-powered insights! 🎉

---

## 📁 What's Connected?

### Frontend → Backend Integration

**API Client:** `Frontend/src/services/weatherApi.js`
- `getWeatherPrediction()` - Main prediction function
- `checkHealth()` - API health check
- Profession-specific helpers

**Config:** `Frontend/src/config/api.js`
- Base URL: http://localhost:8000
- All endpoints configured

**Demo Page:** `Frontend/src/pages/Demo.jsx`
- Live interactive form
- Real-time API integration
- Weather visualization

### Backend API

**Main App:** `AI-Backend/app.py` (FastAPI)
- `/predict` - Weather + AI insights
- `/health` - Status check
- Gemini LLM integration
- TFT model (uses dummy data for demo)

---

## 🛠️ Troubleshooting

### Backend won't start?
```powershell
# Install FastAPI and Uvicorn specifically
pip install fastapi uvicorn python-dotenv
```

### Gemini API Error?
- Check `.env` file has valid `GEMINI_API_KEY`
- Get key: https://aistudio.google.com/app/apikey
- Restart backend after adding key

### Frontend can't connect?
- Check backend is running: http://localhost:8000/health
- Verify `.env` has `VITE_API_URL=http://localhost:8000`
- Restart frontend after changing `.env`

### Port already in use?
```powershell
# Frontend: Use different port
npm run dev -- --port 3001

# Backend: Use different port
uvicorn app:app --reload --port 8001
# Update Frontend .env: VITE_API_URL=http://localhost:8001
```

---

## 📊 Features

### ✅ Frontend (Complete)
- Home page with features
- Features detail page
- About page
- Contact page
- **Demo page (NEW!)** - Live API integration
- Mobile responsive
- Beautiful UI with Tailwind

### ⚙️ Backend (Setup Required)
- FastAPI server
- Weather prediction endpoint
- Gemini AI integration
- Health check
- Interactive API docs
- CORS enabled

---

## 🎯 Next Steps

1. **Run the demo** (http://localhost:3000/demo)
2. **Get Gemini API key** and add to `.env`
3. **Train model** with real data (optional)
4. **Deploy** to production

---

**Full Documentation:** `SETUP_AND_RUN.md`
