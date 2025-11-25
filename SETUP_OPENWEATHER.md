# 🌤️ OpenWeather API Setup Guide

## ⚠️ CRITICAL: Get a Valid OpenWeather API Key

Your current API key is **invalid** or **expired**. Follow these steps:

### Step 1: Create OpenWeather Account

1. Go to https://openweathermap.org/api
2. Click **"Sign Up"** (top right)
3. Fill in your details and create account
4. Verify your email

### Step 2: Generate API Key

1. Log in to https://home.openweathermap.org/api_keys
2. You'll see a default API key already created
3. **OR** create a new one:
   - Enter a name (e.g., "Mausam-Vaani")
   - Click **"Generate"**
4. **Copy the API key** (long string like: `abc123def456...`)

### Step 3: Wait for Activation ⏳

**IMPORTANT:** New API keys take **10-15 minutes** to activate!

- Don't use it immediately after creation
- Wait at least 15 minutes
- During this time you'll get 401 errors (this is normal)

### Step 4: Update Your .env File

Open `AI-Backend/.env` and replace the old key:

```env
OPENWEATHER_API_KEY=your_new_key_here
```

### Step 5: Restart Backend

```powershell
# Stop the server (Ctrl+C)
# Restart it
cd AI-Backend
python app.py
```

---

## ✅ How to Verify It's Working

Once your API key is active (after 15 minutes):

### Test 1: Check Backend Logs

```powershell
python app.py
```

You should see:
```
✓ Using real weather data from OpenWeatherMap for Delhi
```

**NOT:**
```
⚠️ Using dummy data (OpenWeather API unavailable)
OpenWeather API error: 401
```

### Test 2: Test from Frontend

1. Go to http://localhost:3000/demo
2. Enter "Delhi" and click "Get Weather Prediction"
3. Check backend terminal - should show:
   ```
   ✓ Using real weather data from OpenWeatherMap for Delhi
   🤖 Generating AI insights with Gemini...
   ```

---

## 📋 Free Plan Limits

OpenWeather **Free Plan** includes:

- ✅ Current Weather Data
- ✅ 5 Day / 3 Hour Forecast
- ✅ 60 calls/minute
- ✅ 1,000,000 calls/month
- ❌ No historical data (we create synthetic historical data from current conditions)

**This is perfect for your demo!**

---

## 🔧 Current Implementation Flow

```
Frontend (Location Name)
    ↓
Backend receives "Delhi"
    ↓
OpenWeather Geocoding API → Get lat/lon for "Delhi"
    ↓
OpenWeather Current Weather API → Get real-time conditions
    ↓
Create 168hrs synthetic historical data (based on current conditions)
    ↓
TFT Deep Learning Model → Predict next 24hrs
    ↓
Gemini LLM → Analyze predictions + Generate personalized insights
    ↓
Return to Frontend → Display results
```

---

## 🐛 Troubleshooting

### Error: "401 Unauthorized"

**Cause:** API key invalid, expired, or not activated yet

**Solution:**
1. Check you copied the key correctly (no spaces)
2. Wait 15 minutes after creating new key
3. Verify key at https://home.openweathermap.org/api_keys

### Error: "404 Not Found"

**Cause:** Location name not recognized

**Solution:** 
- Use common city names ("Delhi", "Mumbai", "Bangalore")
- Not village names or very specific locations

### Still Using Dummy Data

**Cause:** OpenWeather API call failing

**Solution:**
1. Check internet connection
2. Verify API key in `.env` file
3. Check backend logs for exact error message
4. Test API key manually:
   ```
   https://api.openweathermap.org/data/2.5/weather?q=Delhi&appid=YOUR_KEY
   ```

---

## 📞 Support

- OpenWeather Docs: https://openweathermap.org/api
- OpenWeather Support: https://openweathermap.org/faq
- API Status: https://status.openweathermap.org/

---

## ✨ What You Get with Real Data

### Without OpenWeather (Dummy Data):
- ❌ Same predictions every time
- ❌ Not realistic
- ❌ Can't show actual current conditions
- ✅ Works for basic testing

### With OpenWeather (Real Data):
- ✅ Real-time current weather
- ✅ Location-specific data
- ✅ Accurate temperature, humidity, wind, etc.
- ✅ Better AI predictions
- ✅ Gemini gets real context for better insights
- ✅ Professional demo quality

---

**Get your API key now and make Mausam-Vaani truly intelligent!** 🚀
