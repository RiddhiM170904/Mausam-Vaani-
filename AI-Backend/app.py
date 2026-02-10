"""
🌤️ MAUSAM-VAANI - PRODUCTION FASTAPI SERVER
Complete weather prediction + LLM insights API

This single file includes:
- FastAPI server with CORS
- TFT model loading and prediction
- Gemini LLM integration for personalized insights
- Health check endpoints
- Error handling

Just run: uvicorn app:app --reload
"""

import os
import sys
import json
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
import requests

# FastAPI
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Gemini AI
import google.generativeai as genai

# Environment variables
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'fast_weather_model.pth')  # or 'best_weather_model.pth'
MODEL_TYPE = os.getenv('MODEL_TYPE', 'fast')  # 'fast' or 'slow'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found in environment variables!")
if not OPENWEATHER_API_KEY:
    logger.warning("⚠️ OPENWEATHER_API_KEY not found in environment variables!")

# Model parameters - will be auto-detected from checkpoint
MODEL_CONFIG = {
    'fast': {
        'num_features': 20,
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'forecast_horizon': 12,  # Base trained length, can predict more
        'output_dim': 13,
        'dropout': 0.1,
        'encoder_steps': 72,
    },
    'slow': {
        'num_features': 23,
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 3,
        'forecast_horizon': 24,  # Base trained length, can predict more
        'output_dim': 13,
        'dropout': 0.15,
        'encoder_steps': 168,
    }
}

# Try to load model metadata if available
METADATA_PATH = os.getenv('METADATA_PATH', 'fast_model_metadata.pkl')  # or 'model_metadata.pkl'
model_metadata = None
input_features = []
target_features = []

if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, 'rb') as f:
            model_metadata = pickle.load(f)
        logger.info(f"✓ Loaded model metadata from {METADATA_PATH}")
        input_features = model_metadata.get('input_features', [])
        target_features = model_metadata.get('target_features', [])
        logger.info(f"  Input features ({len(input_features)}): {input_features}")
        logger.info(f"  Target features ({len(target_features)}): {target_features}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load metadata: {e}")

# Fallback feature definitions if metadata not available
if not input_features:
    input_features = [
        'temperature', 'humidity', 'wind_speed', 'rainfall',
        'pressure', 'cloud_cover', 'aqi', 'pm25', 'pm10',
        'co', 'no2', 'o3', 'so2', 'latitude', 'longitude',
        'hour_sin', 'hour_cos', 'day_of_week', 'month', 'is_weekend'
    ]
if not target_features:
    target_features = [
        'temperature', 'humidity', 'wind_speed', 'rainfall',
        'pressure', 'cloud_cover', 'aqi', 'pm25', 'pm10',
        'co', 'no2', 'o3', 'so2'
    ]

# ============================================================================
# GLOBAL MODEL INSTANCE AND DEVICE
# ============================================================================

# Global variables for model and device
weather_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️ Using device: {device}")

# ============================================================================
# MODEL LOADING FUNCTION (Define before lifespan)
# ============================================================================

def load_model():
    """Load the trained TFT model"""
    global weather_model
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"⚠️ Model file not found: {MODEL_PATH}")
            logger.info("✓ API will use dummy data for predictions (perfect for demo!)")
            return False
        
        # Create model instance
        config = MODEL_CONFIG[MODEL_TYPE]
        weather_model = TemporalFusionTransformer(
            num_features=config['num_features'],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            forecast_horizon=config['forecast_horizon'],
            output_dim=config['output_dim'],
            dropout=config['dropout']
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Training checkpoint format
            state_dict = checkpoint['model_state_dict']
            logger.info(f"Loading from training checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # Raw model state dict
            state_dict = checkpoint

        try:
            weather_model.load_state_dict(state_dict)
            weather_model.eval()
            logger.info(f"✓ Model loaded successfully on {device}")
            return True
        except Exception as e:
            # Detailed diagnostics to help users match checkpoint vs model
            logger.warning(f"⚠️ Could not load model strictly: {e}")

            # Print shapes of checkpoint params
            try:
                ckpt_keys = list(state_dict.keys())
                logger.info(f"Checkpoint contains {len(ckpt_keys)} parameter tensors")
                for k in ckpt_keys[:20]:
                    v = state_dict[k]
                    logger.debug(f"CKPT {k}: {tuple(v.shape) if hasattr(v, 'shape') else type(v)}")
            except Exception:
                logger.debug("Unable to enumerate checkpoint parameter shapes")

            # Print model parameter shapes
            try:
                model_keys = [n for n, p in weather_model.named_parameters()]
                logger.info(f"Model expects {len(model_keys)} parameter tensors")
                for i, (n, p) in enumerate(weather_model.named_parameters()):
                    if i < 20:
                        logger.debug(f"MODEL {n}: {tuple(p.shape)}")
            except Exception:
                logger.debug("Unable to enumerate model parameter shapes")

            # Optionally allow a partial (non-strict) load when explicitly requested
            force_partial = os.getenv('FORCE_PARTIAL_MODEL_LOAD', 'false').lower() in ['1', 'true', 'yes']
            if force_partial:
                try:
                    weather_model.load_state_dict(state_dict, strict=False)
                    weather_model.eval()
                    logger.warning('⚠️ Model partially loaded with strict=False. Weights mismatches were ignored.')
                    logger.warning('⚠️ Predictions may be incorrect. Prefer re-saving model with matching architecture.')
                    return True
                except Exception as e2:
                    logger.error(f"Partial load also failed: {e2}")

            logger.info("✓ API will use dummy data for predictions (perfect for demo!)")
            # Reset model to None to avoid using a mismatched model
            weather_model = None
            return False
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load model: {e}")
        logger.info("✓ API will use dummy data for predictions (perfect for demo!)")
        weather_model = None
        return False

# ============================================================================
# LIFESPAN EVENT HANDLER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Startup
    logger.info("🚀 Starting Mausam-Vaani API server...")
    success = load_model()
    if not success:
        logger.warning("Model not loaded - using dummy data for predictions")
    logger.info(f"Device: {device}")
    logger.info(f"Gemini API: {'Configured' if GEMINI_API_KEY else 'Not configured'}")
    yield
    # Shutdown
    logger.info("Shutting down Mausam-Vaani API server...")

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Mausam-Vaani Weather API",
    description="Hyperlocal weather prediction with AI-powered personalized insights",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# TFT MODEL DEFINITION (Must match training script)
# ============================================================================

class SimpleTFT(nn.Module):
    """Fast, simplified Temporal Fusion Transformer (from colab_fast_train.py)"""
    
    def __init__(self, num_features, hidden_dim, num_heads, num_layers,
                 forecast_horizon, output_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Encoder LSTM (unidirectional for speed)
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output layers
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, encoder_steps=72, forecast_steps=12):
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Encode historical data
        encoder_input = x[:, :encoder_steps, :]
        encoded, (h, c) = self.encoder_lstm(encoder_input)
        
        # Self-attention on encoder output
        attn_out, _ = self.self_attention(encoded, encoded, encoded)
        encoded = self.layer_norm(encoded + attn_out)
        
        # Decode future
        decoder_input = torch.zeros(batch_size, forecast_steps, self.hidden_dim).to(x.device)
        decoded, _ = self.decoder_lstm(decoder_input, (h, c))
        
        # Gated output
        gate = torch.sigmoid(self.output_gate(decoded))
        output = decoded * gate
        output = self.dropout(output)
        
        # Final projection
        predictions = self.output_proj(output)
        
        return predictions

# Alias for compatibility
TemporalFusionTransformer = SimpleTFT

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class WeatherInput(BaseModel):
    """Input for weather prediction"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "location_name": "Delhi"
            }
        }
    )
    
    location_name: str = Field(..., description="Location name (e.g., 'Delhi', 'Mumbai')")
    latitude: Optional[float] = Field(None, description="Latitude (optional, will be fetched from API)", ge=-90, le=90)
    longitude: Optional[float] = Field(None, description="Longitude (optional, will be fetched from API)", ge=-180, le=180)

class UserContext(BaseModel):
    """User context for personalized insights"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "profession": "Farmer",
                "additional_context": {"crop": "Rice"}
            }
        }
    )
    
    profession: str = Field(
        "General", 
        description="User profession (Farmer, Commuter, Construction Worker, Outdoor Sports, General)"
    )
    additional_context: Optional[Dict] = Field(
        None,
        description="Additional context (e.g., {'crop': 'Rice', 'vehicle': 'Bike'})"
    )

class PredictionRequest(BaseModel):
    """Complete prediction request"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "weather_input": {
                    "location_name": "Delhi"
                },
                "user_context": {
                    "profession": "Farmer",
                    "additional_context": {"crop": "Rice"}
                },
                "forecast_hours": 24
            }
        }
    )
    
    weather_input: WeatherInput
    user_context: UserContext
    forecast_hours: int = Field(24, description="Number of hours to forecast (frontend controlled)", ge=1)

class WeatherPrediction(BaseModel):
    """Weather prediction output with all parameters"""
    timestamp: str
    temperature: float
    humidity: float
    wind_speed: float
    rainfall: float
    pressure: float
    cloud_cover: float
    aqi: Optional[float] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    co: Optional[float] = None
    no2: Optional[float] = None
    o3: Optional[float] = None
    so2: Optional[float] = None
    sunrise: Optional[str] = None
    sunset: Optional[str] = None
    moonrise: Optional[str] = None
    moonset: Optional[str] = None
    moon_phase: Optional[str] = None

class InsightResponse(BaseModel):
    """Complete API response"""
    location: str
    latitude: float
    longitude: float
    current_time: str
    forecast_hours: int
    current_weather: Dict  # Real-time data from OpenWeather
    predictions: List[WeatherPrediction]  # AI predictions refined by Gemini
    summary: Dict[str, float]  # avg, min, max for key metrics
    personalized_insight: str
    profession: str
    model_used: str  # 'fast' or 'slow'
    data_source: str  # 'model' or 'demo'

# ============================================================================
# GEMINI LLM INTEGRATION
# ============================================================================

def generate_gemini_insight(weather_data: Dict, user_context: UserContext) -> str:
    """Generate personalized insight using Gemini LLM"""
    
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not configured, using fallback insights")
        return generate_fallback_insight(weather_data, user_context)
    
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build prompt
        prompt = build_gemini_prompt(weather_data, user_context)
        
        # Generate insight
        logger.info("Calling Gemini API for personalized insight...")
        response = model.generate_content(prompt)
        insight = response.text.strip()
        
        logger.info("✓ Gemini insight generated successfully")
        return insight
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return generate_fallback_insight(weather_data, user_context)

def build_gemini_prompt(weather_data: Dict, user_context: UserContext) -> str:
    """Build comprehensive prompt for Gemini with current + predicted weather analysis"""
    
    location = weather_data.get('location', 'your location')
    current_weather = weather_data.get('current_weather', {})
    predictions = weather_data.get('predictions', [])
    summary = weather_data['summary']
    
    # Extract current conditions
    current_temp = current_weather.get('temp', summary['avg_temperature'])
    current_humidity = current_weather.get('humidity', summary['avg_humidity'])
    current_wind = current_weather.get('wind_speed', summary['avg_wind_speed'])
    current_rainfall = current_weather.get('rainfall', 0)
    current_pressure = current_weather.get('pressure', 1010)
    current_cloud = current_weather.get('cloud_cover', 50)
    
    # Extract prediction summary
    avg_temp = summary['avg_temperature']
    min_temp = summary['min_temperature']
    max_temp = summary['max_temperature']
    total_rainfall = summary['total_rainfall']
    avg_humidity = summary['avg_humidity']
    avg_wind = summary['avg_wind_speed']
    
    # Analyze trends from predictions
    temps = [p['temperature'] if isinstance(p, dict) else p.temperature for p in predictions[:6]] if predictions else [avg_temp]
    temp_trend = "rising" if temps[-1] > temps[0] else "falling" if temps[-1] < temps[0] else "stable"
    
    # Detect current season
    current_month = datetime.now().month
    if current_month in [6, 7, 8, 9]:
        season = "Monsoon"
    elif current_month in [11, 12, 1, 2]:
        season = "Winter"
    elif current_month in [3, 4, 5]:
        season = "Summer"
    else:
        season = "Spring"
    
    # Determine overall condition (prioritize temperature in dry seasons)
    if season == "Winter":
        # Winter: prioritize temperature, rain is rare
        if avg_temp < 10:
            condition = "Very Cold"
            condition_emoji = "❄️"
        elif avg_temp < 18:
            condition = "Cold & Pleasant"
            condition_emoji = "🥶"
        elif total_rainfall > 20:  # Higher threshold for winter
            condition = "Unexpected Rain"
            condition_emoji = "🌧️"
        else:
            condition = "Pleasant Winter Day"
            condition_emoji = "☀️"
    elif season == "Summer":
        # Summer: heat is primary concern
        if avg_temp > 40:
            condition = "Extreme Heat"
            condition_emoji = "🔥"
        elif avg_temp > 35:
            condition = "Very Hot"
            condition_emoji = "🌡️"
        elif total_rainfall > 10:
            condition = "Hot with Rain"
            condition_emoji = "🌦️"
        else:
            condition = "Warm & Dry"
            condition_emoji = "☀️"
    elif season == "Monsoon":
        # Monsoon: rain is expected
        if total_rainfall > 50:
            condition = "Heavy Monsoon Rain"
            condition_emoji = "🌧️"
        elif total_rainfall > 20:
            condition = "Moderate Rain"
            condition_emoji = "🌦️"
        elif total_rainfall > 5:
            condition = "Light Showers"
            condition_emoji = "☔"
        else:
            condition = "Dry Spell"
            condition_emoji = "🌤️"
    else:
        # Spring: balanced
        if total_rainfall > 30:
            condition = "Rainy"
            condition_emoji = "🌧️"
        elif avg_temp > 30:
            condition = "Warm"
            condition_emoji = "🌤️"
        elif avg_temp < 15:
            condition = "Cool"
            condition_emoji = "🥶"
        else:
            condition = "Pleasant"
            condition_emoji = "😊"
    
    profession = user_context.profession
    context = user_context.additional_context or {}
    
    # Extract user planning details
    planned_activity = context.get('planned_activity', '')
    activity_time = context.get('activity_time', '')
    duration = context.get('duration', '')
    concerns = context.get('specific_concerns', '')
    location_type = context.get('location_type', 'City')
    village = context.get('village', '')
    district = context.get('district', '')
    
    # Build comprehensive weather analysis prompt
    prompt = f"""You are Mausam-Vaani, a friendly AI weather advisor for India 🇮🇳. Provide warm, personalized advice in a conversational tone.

📍 LOCATION: {location}
{f"🏘️ Village: {village}, District: {district}" if village else f"🏛️ District/City: {district}" if district else ""}
{f"📌 Hyperlocal Precision: Village-level weather analysis" if location_type == 'Village' else ""}

🌍 SEASON: {season}

🌡️ CURRENT CONDITIONS (Right Now):
• Temperature: {current_temp:.1f}°C
• Feels Like: {current_temp + (current_humidity - 50) * 0.1:.1f}°C (humidity adjusted)
• Humidity: {current_humidity:.0f}%
• Wind: {current_wind:.1f} km/h
• Rainfall: {current_rainfall:.1f} mm/h {"☔ (Raining now!)" if current_rainfall > 0 else "✅ (No rain)"}
• Pressure: {current_pressure:.0f} hPa
• Cloud Cover: {current_cloud:.0f}% {"☁️" if current_cloud > 70 else "🌤️" if current_cloud > 30 else "☀️"}

📊 AI PREDICTIONS (Next Hours):
• Condition: {condition_emoji} {condition}
• Temperature: {min_temp:.1f}°C to {max_temp:.1f}°C (Trend: {temp_trend})
• Expected Rainfall: {total_rainfall:.1f} mm {"(Typical for monsoon)" if season == "Monsoon" else "(Rare for this season)" if total_rainfall > 5 and season == "Winter" else ""}
• Humidity: {avg_humidity:.0f}%
• Wind: {avg_wind:.1f} km/h

👤 ABOUT YOU:
• Role/Occupation: {profession}
{f"• Planned Activity: {planned_activity}" if planned_activity else ""}
{f"• When: {activity_time}" if activity_time else ""}
{f"• Duration: {duration}" if duration else ""}
{f"• Your Concerns: {concerns}" if concerns else ""}

💡 YOUR TASK:
Provide friendly, actionable weather advice specifically for this person's situation. Consider:
1. Current SEASON ({season}) - don't over-emphasize rain if it's winter/summer dry season
2. Their location (hyperlocal village or city)
3. Their planned activity and timing
4. Current weather + upcoming changes
5. Any specific concerns they mentioned

Write in a warm, conversational tone like talking to a friend. Use emojis naturally. Be specific about timing and practical actions.

IMPORTANT: If total rainfall is less than 2mm and it's winter/summer, DON'T focus on rain - talk about temperature, wind, sun, or other relevant factors instead!

Format your response as:
[Greeting + Weather Overview] → [Specific Recommendations] → [Safety/Tips] → [Encouragement]

Keep it 3-4 sentences, friendly and helpful!"""
    
    return prompt

def generate_fallback_insight(weather_data: Dict, user_context: UserContext) -> str:
    """Generate basic insight when Gemini is unavailable"""
    
    avg_temp = weather_data['summary']['avg_temperature']
    total_rainfall = weather_data['summary']['total_rainfall']
    profession = user_context.profession
    
    insights = []
    
    # Temperature-based
    if avg_temp > 35:
        insights.append("🌡️ Very hot weather expected. Stay hydrated and avoid outdoor activities during peak hours (12-4 PM).")
    elif avg_temp < 15:
        insights.append("🥶 Cold weather ahead. Wear warm clothing.")
    
    # Rainfall-based
    if total_rainfall > 50:
        insights.append("🌧️ Heavy rain expected. Carry umbrella and avoid waterlogged areas.")
    elif total_rainfall > 10:
        insights.append("☔ Moderate rain expected. Plan accordingly and carry rain gear.")
    elif total_rainfall > 0:
        insights.append("🌦️ Light rain possible. Keep an umbrella handy.")
    
    # Profession-specific
    if profession == 'Farmer':
        if total_rainfall > 10:
            insights.append("🚫 Postpone pesticide spraying and outdoor harvesting due to rain.")
        else:
            insights.append("✅ Good conditions for regular farming activities.")
    elif profession == 'Commuter':
        if total_rainfall > 0:
            insights.append("🚗 Expect traffic delays due to rain. Plan extra travel time.")
    elif profession == 'Construction Worker':
        if total_rainfall > 0:
            insights.append("⚠️ Halt outdoor construction work during rain.")
        elif avg_temp > 35:
            insights.append("💧 Take frequent breaks and stay hydrated in hot weather.")
    
    # Default
    if not insights:
        insights.append("✅ Weather conditions are favorable for regular activities.")
    
    return " ".join(insights[:3])

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_openweather_data(lat: float, lon: float, location_name: str = None) -> Dict:
    """
    Fetch real-time and forecast weather data from OpenWeatherMap API
    Returns historical-like data + current conditions + coordinates
    """
    if not OPENWEATHER_API_KEY:
        logger.warning("OpenWeather API key not configured, using dummy data")
        return None
    
    try:
        # Use provided lat/lon from frontend (captured location)
        logger.info(f"Fetching weather for coordinates: ({lat}, {lon})")
        
        # Get location name from reverse geocoding if not provided
        if not location_name:
            geo_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHER_API_KEY}"
            geo_response = requests.get(geo_url, timeout=10)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            location_name = geo_data[0].get('name', f"Lat:{lat}, Lon:{lon}") if geo_data else f"Lat:{lat}, Lon:{lon}"
        
        actual_name = location_name
        logger.info(f"Location: {actual_name} ({lat}, {lon})")
        
        # Get current weather + 5 day/3 hour forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        # Get current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        current_response = requests.get(current_url, timeout=10)
        current_response.raise_for_status()
        current_data = current_response.json()
        
        # Get astronomical data (sunrise, sunset)
        sunrise_ts = current_data['sys'].get('sunrise', 0)
        sunset_ts = current_data['sys'].get('sunset', 0)
        sunrise_time = datetime.fromtimestamp(sunrise_ts).strftime('%H:%M') if sunrise_ts else 'N/A'
        sunset_time = datetime.fromtimestamp(sunset_ts).strftime('%H:%M') if sunset_ts else 'N/A'
        
        # Build historical-like data from current + forecast
        historical_data = []
        
        # Add current weather as most recent point
        current_weather = {
            'temp': current_data['main']['temp'],
            'humidity': current_data['main']['humidity'],
            'wind_speed': current_data['wind']['speed'] * 3.6,  # m/s to km/h
            'rainfall': current_data.get('rain', {}).get('1h', 0),
            'pressure': current_data['main']['pressure'],
            'cloud_cover': current_data['clouds']['all'],
            'sunrise': sunrise_time,
            'sunset': sunset_time,
            'description': current_data['weather'][0].get('description', 'Unknown'),
        }
        
        # Create 168 hours of synthetic historical data based on current conditions
        # (In production, you'd fetch actual historical data from a paid API)
        base_temp = current_weather['temp']
        base_humidity = current_weather['humidity']
        base_wind = current_weather['wind_speed']
        base_pressure = current_weather['pressure']
        base_cloud = current_weather['cloud_cover']
        
        # Seasonal awareness
        current_month = datetime.now().month
        is_monsoon = current_month in [6, 7, 8, 9]
        is_winter = current_month in [11, 12, 1, 2]
        
        for hour in range(168):
            hour_of_day = (datetime.now().hour - (168 - hour)) % 24
            
            # Daily temperature variation
            temp_variation = 5 * np.sin((hour_of_day - 6) * np.pi / 12)
            temp = base_temp + temp_variation + np.random.randn() * 2
            
            humidity = max(30, min(95, base_humidity - temp_variation * 2 + np.random.randn() * 5))
            wind_speed = max(0, base_wind + np.random.randn() * 2)
            
            # Seasonal rainfall
            if is_monsoon:
                rainfall = max(0, np.random.randn() * 4) if np.random.rand() > 0.70 else 0
            elif is_winter:
                rainfall = max(0, np.random.randn() * 0.3) if np.random.rand() > 0.96 else 0
            else:
                rainfall = max(0, np.random.randn() * 2) if np.random.rand() > 0.90 else 0
            
            pressure = base_pressure + np.random.randn() * 3
            cloud = max(0, min(100, base_cloud + np.random.randn() * 15))
            
            historical_data.append([temp, humidity, wind_speed, rainfall, pressure, cloud, lat, lon, hour_of_day / 24.0])
        
        # Replace last entry with actual current data
        current_hour = datetime.now().hour
        historical_data[-1] = [
            current_weather['temp'],
            current_weather['humidity'],
            current_weather['wind_speed'],
            current_weather['rainfall'],
            current_weather['pressure'],
            current_weather['cloud_cover'],
            lat, lon, current_hour / 24.0
        ]
        
        logger.info(f"✓ Fetched real weather data for {actual_name}")
        
        return {
            'location_name': actual_name,
            'latitude': lat,
            'longitude': lon,
            'historical_data': np.array(historical_data),
            'current_weather': current_weather
        }
        
    except requests.RequestException as e:
        error_msg = str(e)
        if '401' in error_msg:
            logger.error(f"OpenWeather API error: Invalid API key. Please get a valid key from https://openweathermap.org/api")
            logger.error(f"Note: New API keys can take 10-15 minutes to activate after creation")
        else:
            logger.error(f"OpenWeather API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return None

def create_dummy_historical_data(lat: float, lon: float) -> np.ndarray:
    """Create dummy historical data for demo purposes"""
    # 168 hours (7 days) of hourly data
    # Features: [temp, humidity, wind_speed, rainfall, pressure, cloud_cover, lat, lon, hour]
    
    np.random.seed(int(lat * 1000 + lon * 1000))
    
    data = []
    base_temp = 25 + (lat - 20) * 0.5  # Temperature varies with latitude
    
    for hour in range(168):
        # Simulate daily temperature cycle
        hour_of_day = hour % 24
        temp_variation = 5 * np.sin((hour_of_day - 6) * np.pi / 12)
        temp = base_temp + temp_variation + np.random.randn() * 2
        
        humidity = 60 + np.random.randn() * 10
        wind_speed = 5 + np.random.randn() * 2
        rainfall = max(0, np.random.randn() * 2) if np.random.rand() > 0.8 else 0
        pressure = 1010 + np.random.randn() * 5
        cloud_cover = 50 + np.random.randn() * 20
        
        # 9 features: temp, humidity, wind, rainfall, pressure, cloud, lat, lon, hour_of_day
        data.append([temp, humidity, wind_speed, rainfall, pressure, cloud_cover, lat, lon, hour_of_day / 24.0])
    
    return np.array(data)

def make_prediction(input_data: np.ndarray, forecast_hours: int) -> np.ndarray:
    """Make weather prediction using the TFT model for any number of hours"""
    
    # If model is not loaded, use realistic dummy predictions
    if weather_model is None:
        logger.info(f"Using dummy data for {forecast_hours}h prediction (model not loaded)")
        try:
            logger.debug(f"Input data shape: {getattr(input_data, 'shape', None)}")
            last_vals = input_data[-1] if len(input_data) > 0 else None
            logger.debug(f"Last historical values sample: {last_vals}")
        except Exception:
            logger.debug("Unable to inspect input_data for debug")
        return generate_realistic_dummy_prediction(input_data, forecast_hours)
    
    try:
        # Use encoder steps from model config
        config = MODEL_CONFIG[MODEL_TYPE]
        encoder_steps = config['encoder_steps']
        total_steps = encoder_steps + forecast_hours
        
        # Pad or trim input data
        if len(input_data) < total_steps:
            # Pad with last values
            padding = np.repeat(input_data[-1:], total_steps - len(input_data), axis=0)
            input_data = np.vstack([input_data, padding])
        else:
            input_data = input_data[-total_steps:]
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            predictions = weather_model(
                input_tensor, 
                encoder_steps=encoder_steps, 
                forecast_steps=forecast_hours
            )
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()[0]  # Shape: (forecast_hours, output_dim)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {e}, falling back to dummy data")
        return generate_realistic_dummy_prediction(input_data, forecast_hours)


def generate_realistic_dummy_prediction(input_data: np.ndarray, forecast_hours: int) -> np.ndarray:
    """Generate realistic dummy predictions based on input data"""
    # Use last historical values as base
    last_values = input_data[-1]
    
    # Extract base values (temp, humidity, wind_speed, rainfall, pressure, cloud_cover)
    base_temp = last_values[0]
    base_humidity = last_values[1]
    base_wind = last_values[2]
    base_pressure = last_values[4] if len(last_values) > 4 else 1010.0
    base_cloud = last_values[5] if len(last_values) > 5 else 50.0
    
    # Seasonal awareness (simple heuristic based on current month)
    current_month = datetime.now().month
    # Monsoon: Jun-Sep (6-9), Winter: Nov-Feb (11,12,1,2), Summer: Mar-May (3-5)
    is_monsoon = current_month in [6, 7, 8, 9]
    is_winter = current_month in [11, 12, 1, 2]
    
    predictions = []
    for hour in range(forecast_hours):
        # Add realistic variations
        hour_of_day = (hour % 24)
        
        # Temperature: daily cycle + small random variation
        temp_variation = 5 * np.sin((hour_of_day - 6) * np.pi / 12)
        temp = base_temp + temp_variation + np.random.randn() * 1.5
        
        # Humidity: inverse of temperature
        humidity = max(30, min(95, base_humidity - temp_variation * 2 + np.random.randn() * 5))
        
        # Wind speed: varies slightly
        wind = max(0, base_wind + np.random.randn() * 2)
        
        # Rainfall: seasonal and realistic
        if is_monsoon:
            # Monsoon: higher chance, more rain
            rainfall = max(0, np.random.randn() * 5) if np.random.rand() > 0.70 else 0
        elif is_winter:
            # Winter: very rare rain, minimal amounts
            rainfall = max(0, np.random.randn() * 0.5) if np.random.rand() > 0.95 else 0
        else:
            # Summer/Spring: occasional light rain
            rainfall = max(0, np.random.randn() * 2) if np.random.rand() > 0.90 else 0
        
        # Pressure: slight variations
        pressure = base_pressure + np.random.randn() * 2
        
        # Cloud cover: loosely correlated with rainfall, but not extreme
        cloud = min(100, max(0, base_cloud + (rainfall * 3) + np.random.randn() * 10))
        
        predictions.append([temp, humidity, wind, rainfall, pressure, cloud])
    
    preds = np.array(predictions)
    try:
        logger.debug(f"Generated dummy predictions shape: {preds.shape}")
        logger.debug(f"Dummy predictions (first 3 rows): {preds[:3].tolist()}")
    except Exception:
        logger.debug("Unable to log dummy predictions")
    return preds


def refine_predictions_with_gemini(model_predictions: List[Dict], current_weather: Dict, location: str, lat: float, lon: float) -> List[Dict]:
    """
    Send model predictions + real-time data to Gemini for refinement
    Returns refined exact values for all weather parameters including AQI
    """
    if not GEMINI_API_KEY:
        logger.warning("Gemini API not configured, using raw predictions")
        return model_predictions
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Build refinement prompt
        prompt = f"""You are a meteorological AI assistant. You have:

1. CURRENT REAL-TIME DATA (from OpenWeather API) for {location} ({lat}, {lon}):
   - Temperature: {current_weather.get('temp', 'N/A')}°C
   - Humidity: {current_weather.get('humidity', 'N/A')}%
   - Wind Speed: {current_weather.get('wind_speed', 'N/A')} km/h
   - Pressure: {current_weather.get('pressure', 'N/A')} hPa
   - Cloud Cover: {current_weather.get('cloud_cover', 'N/A')}%
   - Rainfall: {current_weather.get('rainfall', 0)} mm/h
   - Conditions: {current_weather.get('description', 'N/A')}

2. AI MODEL PREDICTIONS (next {len(model_predictions)} hours):
{chr(10).join([f"   Hour {i+1}: Temp={p.get('temperature', 25):.1f}°C, Humidity={p.get('humidity', 60):.0f}%, Rain={p.get('rainfall', 0):.1f}mm" for i, p in enumerate(model_predictions[:5])])}
   ...

TASK: Refine and correct the AI predictions using the real-time data as a baseline. Output ONLY a JSON array with realistic refined values.

Each prediction should include ALL these fields (use realistic Indian weather values):
- temperature (°C, range: -5 to 50)
- humidity (%, range: 20-100)
- wind_speed (km/h, range: 0-60)
- rainfall (mm, range: 0-100)
- pressure (hPa, range: 980-1040)
- cloud_cover (%, range: 0-100)
- aqi (Air Quality Index, 0-500, typical Indian cities: 100-300, rural: 50-150)
- pm25 (μg/m³, typical: 30-150)
- pm10 (μg/m³, typical: 50-250)
- co (ppm, typical: 0.5-3.0)
- no2 (ppb, typical: 20-80)
- o3 (ppb, typical: 30-70)
- so2 (ppb, typical: 5-40)

IMPORTANT:
- Start from current real-time values
- Make gradual, realistic changes hour by hour
- Consider Indian weather patterns and seasons (month: {datetime.now().month})
- AQI should be realistic for Indian locations
- NO explanations, ONLY valid JSON array

Format:
[
  {{"temperature": 28.5, "humidity": 65, "wind_speed": 12.3, "rainfall": 0, "pressure": 1012, "cloud_cover": 45, "aqi": 150, "pm25": 65, "pm10": 95, "co": 1.2, "no2": 35, "o3": 42, "so2": 15}},
  ...
]"""
        
        logger.info("Sending to Gemini for prediction refinement...")
        response = model.generate_content(prompt)
        refined_text = response.text.strip()
        
        # Extract JSON from response
        import re
        import json
        json_match = re.search(r'\[.*\]', refined_text, re.DOTALL)
        if json_match:
            refined_predictions = json.loads(json_match.group(0))
            logger.info(f"✓ Gemini refined {len(refined_predictions)} predictions")
            return refined_predictions[:len(model_predictions)]  # Match requested length
        else:
            logger.warning("Could not parse Gemini response, using raw predictions")
            return model_predictions
            
    except Exception as e:
        logger.error(f"Gemini refinement error: {e}")
        return model_predictions

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🌤️ Mausam-Vaani Weather API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": weather_model is not None,
        "device": str(device),
        "gemini_enabled": GEMINI_API_KEY is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": weather_model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-diagnostics")
async def model_diagnostics():
    """Get detailed model and checkpoint diagnostics"""
    diagnostics = {
        "model_config": MODEL_CONFIG,
        "model_loaded": weather_model is not None,
        "device": str(device),
        "model_path": MODEL_PATH,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "metadata_loaded": model_metadata is not None,
    }
    
    if model_metadata:
        diagnostics["metadata"] = {
            "feature_cols": model_metadata.get('feature_cols', []),
            "target_cols": model_metadata.get('target_cols', []),
            "num_features": model_metadata.get('num_features'),
            "num_targets": model_metadata.get('num_targets'),
        }
    
    if weather_model is not None:
        # Get model parameter info
        model_params = {}
        for i, (name, param) in enumerate(weather_model.named_parameters()):
            if i < 20:  # First 20 params
                model_params[name] = list(param.shape)
        diagnostics["model_parameters_sample"] = model_params
        diagnostics["total_parameters"] = sum(p.numel() for p in weather_model.parameters())
    
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            ckpt_params = {}
            for i, (name, tensor) in enumerate(checkpoint.items()):
                if i < 20:  # First 20 params
                    ckpt_params[name] = list(tensor.shape) if hasattr(tensor, 'shape') else str(type(tensor))
            diagnostics["checkpoint_parameters_sample"] = ckpt_params
            diagnostics["checkpoint_keys_count"] = len(checkpoint)
        except Exception as e:
            diagnostics["checkpoint_load_error"] = str(e)
    
    return diagnostics

def refine_predictions_with_gemini(model_predictions: List[Dict], current_weather: Dict, location: str, lat: float, lon: float) -> List[Dict]:
    """
    Send model predictions + real-time data to Gemini for refinement
    Returns refined exact values for all weather parameters
    """
    if not GEMINI_API_KEY:
        logger.warning("Gemini API not configured, using raw predictions")
        return model_predictions
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Build refinement prompt
        prompt = f"""You are a meteorological AI assistant. You have:

1. CURRENT REAL-TIME DATA (from OpenWeather API) for {location} ({lat}, {lon}):
   - Temperature: {current_weather.get('temp', 'N/A')}°C
   - Humidity: {current_weather.get('humidity', 'N/A')}%
   - Wind Speed: {current_weather.get('wind_speed', 'N/A')} km/h
   - Pressure: {current_weather.get('pressure', 'N/A')} hPa
   - Cloud Cover: {current_weather.get('cloud_cover', 'N/A')}%
   - Rainfall: {current_weather.get('rainfall', 0)} mm/h

2. AI MODEL PREDICTIONS (next {len(model_predictions)} hours):
{chr(10).join([f"   Hour {i+1}: Temp={p['temperature']:.1f}°C, Humidity={p['humidity']:.0f}%, Rain={p.get('rainfall', 0):.1f}mm" for i, p in enumerate(model_predictions[:5])])}
   ...

TASK: Refine and correct the AI predictions using the real-time data as a baseline. Output ONLY a JSON array with refined values.

Each prediction should include ALL these fields (use realistic Indian weather values):
- temperature (°C)
- humidity (%)
- wind_speed (km/h)
- rainfall (mm)
- pressure (hPa)
- cloud_cover (%)
- aqi (Air Quality Index, 0-500)
- pm25 (μg/m³)
- pm10 (μg/m³)
- co (ppm)
- no2 (ppb)
- o3 (ppb)
- so2 (ppb)

IMPORTANT:
- Start from current real-time values
- Make gradual, realistic changes hour by hour
- Consider Indian weather patterns and seasons
- AQI should be realistic for Indian cities (typically 100-300 in cities, 50-100 in rural areas)
- Output ONLY valid JSON array, no explanations

Format:
[
  {{"temperature": 28.5, "humidity": 65, "wind_speed": 12.3, "rainfall": 0, "pressure": 1012, "cloud_cover": 45, "aqi": 150, "pm25": 65, "pm10": 95, "co": 1.2, "no2": 35, "o3": 42, "so2": 15}},
  ...
]"""
        
        logger.info("Sending to Gemini for prediction refinement...")
        response = model.generate_content(prompt)
        refined_text = response.text.strip()
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', refined_text, re.DOTALL)
        if json_match:
            refined_predictions = json.loads(json_match.group(0))
            logger.info(f"✓ Gemini refined {len(refined_predictions)} predictions")
            return refined_predictions[:len(model_predictions)]  # Match requested length
        else:
            logger.warning("Could not parse Gemini response, using raw predictions")
            return model_predictions
            
    except Exception as e:
        logger.error(f"Gemini refinement error: {e}")
        return model_predictions

@app.post("/predict", response_model=InsightResponse)
async def predict_weather(request: PredictionRequest):
    """
    Main prediction endpoint - uses lat/lon from frontend, combines model + OpenWeather + Gemini
    
    Returns weather predictions + personalized LLM insights
    """
    
    try:
        logger.info(f"Prediction request for {request.weather_input.location_name}")
        
        # Extract request data
        location_input = request.weather_input
        user_context = request.user_context
        forecast_hours = request.forecast_hours  # Use exactly what frontend requests
        
        # Use lat/lon from frontend if provided, otherwise try to get from location name
        if location_input.latitude and location_input.longitude:
            lat = location_input.latitude
            lon = location_input.longitude
            location_name = location_input.location_name
            logger.info(f"Using provided coordinates: {lat}, {lon}")
        else:
            # Fallback: try to geocode from location name
            logger.info(f"No coordinates provided, geocoding '{location_input.location_name}'...")
            if not OPENWEATHER_API_KEY:
                raise HTTPException(status_code=400, detail="Location coordinates required when OpenWeather API is not configured")
            
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_input.location_name}&limit=1&appid={OPENWEATHER_API_KEY}"
            geo_response = requests.get(geo_url, timeout=10)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if not geo_data:
                raise HTTPException(status_code=404, detail=f"Location '{location_input.location_name}' not found")
            
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
            location_name = geo_data[0].get('name', location_input.location_name)
        
        # Step 1: Fetch real-time data from OpenWeather API
        logger.info(f"Step 1: Fetching real-time data for {location_name} ({lat}, {lon})...")
        weather_data = fetch_openweather_data(lat, lon, location_name)
        
        if not weather_data:
            logger.warning("OpenWeather unavailable, using dummy data")
            # Create dummy data
            historical_input = create_dummy_historical_data(lat, lon)
            current_weather = {
                'temp': 25.0,
                'humidity': 60.0,
                'wind_speed': 10.0,
                'rainfall': 0.0,
                'pressure': 1012.0,
                'cloud_cover': 50.0,
                'sunrise': '06:00',
                'sunset': '18:00',
                'description': 'Clear sky'
            }
            location_name = location_input.location_name or f"Lat:{lat}, Lon:{lon}"
            data_source = "demo"
        else:
            historical_input = weather_data['historical_data']
            current_weather = weather_data['current_weather']
            location_name = weather_data['location_name']
            lat = weather_data['latitude']
            lon = weather_data['longitude']
            data_source = "openweather+model"
        
        # Step 2: Make AI model predictions
        logger.info(f"Step 2: Making {forecast_hours}h predictions using {MODEL_TYPE} model...")
        raw_predictions = make_prediction(historical_input, forecast_hours)
        
        # Convert raw predictions to dict format
        model_predictions = []
        for i in range(forecast_hours):
            pred_dict = {
                'temperature': float(raw_predictions[i, 0]) if len(raw_predictions[i]) > 0 else 25.0,
                'humidity': float(raw_predictions[i, 1]) if len(raw_predictions[i]) > 1 else 60.0,
                'wind_speed': float(raw_predictions[i, 2]) if len(raw_predictions[i]) > 2 else 10.0,
                'rainfall': float(raw_predictions[i, 3]) if len(raw_predictions[i]) > 3 else 0.0,
                'pressure': float(raw_predictions[i, 4]) if len(raw_predictions[i]) > 4 else 1012.0,
                'cloud_cover': float(raw_predictions[i, 5]) if len(raw_predictions[i]) > 5 else 50.0,
                'aqi': float(raw_predictions[i, 6]) if len(raw_predictions[i]) > 6 else 100.0,
                'pm25': float(raw_predictions[i, 7]) if len(raw_predictions[i]) > 7 else 50.0,
                'pm10': float(raw_predictions[i, 8]) if len(raw_predictions[i]) > 8 else 80.0,
                'co': float(raw_predictions[i, 9]) if len(raw_predictions[i]) > 9 else 1.0,
                'no2': float(raw_predictions[i, 10]) if len(raw_predictions[i]) > 10 else 30.0,
                'o3': float(raw_predictions[i, 11]) if len(raw_predictions[i]) > 11 else 40.0,
                'so2': float(raw_predictions[i, 12]) if len(raw_predictions[i]) > 12 else 10.0,
            }
            model_predictions.append(pred_dict)
        
        # Step 3: Refine predictions with Gemini using real-time data
        logger.info(f"Step 3: Refining predictions with Gemini AI...")
        refined_predictions = refine_predictions_with_gemini(
            model_predictions, current_weather, location_name, lat, lon
        )
        
        # Step 4: Create WeatherPrediction objects with timestamps and astronomical data
        predictions = []
        current_time = datetime.now()
        
        # Calculate sunrise/sunset for each hour (simplified - use current day's times)
        sunrise = current_weather.get('sunrise', '06:00')
        sunset = current_weather.get('sunset', '18:00')
        
        for i, pred in enumerate(refined_predictions):
            forecast_time = current_time + timedelta(hours=i+1)
            
            # Simple moon phase calculation (approximate)
            day_of_month = forecast_time.day
            moon_phase = "New Moon" if day_of_month < 4 else "First Quarter" if day_of_month < 11 else "Full Moon" if day_of_month < 19 else "Last Quarter" if day_of_month < 26 else "New Moon"
            
            predictions.append(WeatherPrediction(
                timestamp=forecast_time.strftime('%Y-%m-%d %H:%M'),
                temperature=pred.get('temperature', 25.0),
                humidity=pred.get('humidity', 60.0),
                wind_speed=pred.get('wind_speed', 10.0),
                rainfall=pred.get('rainfall', 0.0),
                pressure=pred.get('pressure', 1012.0),
                cloud_cover=pred.get('cloud_cover', 50.0),
                aqi=pred.get('aqi', 100.0),
                pm25=pred.get('pm25', 50.0),
                pm10=pred.get('pm10', 80.0),
                co=pred.get('co', 1.0),
                no2=pred.get('no2', 30.0),
                o3=pred.get('o3', 40.0),
                so2=pred.get('so2', 10.0),
                sunrise=sunrise if i == 0 else None,  # Only show for first hour
                sunset=sunset if i == 0 else None,
                moonrise=None,  # Would need additional API call
                moonset=None,
                moon_phase=moon_phase if i == 0 else None
            ))
        
        # Step 5: Calculate summary statistics
        temps = [p.temperature for p in predictions]
        humidities = [p.humidity for p in predictions]
        wind_speeds = [p.wind_speed for p in predictions]
        rainfalls = [p.rainfall for p in predictions]
        
        summary = {
            'avg_temperature': float(np.mean(temps)),
            'min_temperature': float(np.min(temps)),
            'max_temperature': float(np.max(temps)),
            'avg_humidity': float(np.mean(humidities)),
            'avg_wind_speed': float(np.mean(wind_speeds)),
            'total_rainfall': float(np.sum(rainfalls)),
        }
        
        # Step 6: Generate personalized insight with Gemini
        logger.info("Step 4: Generating personalized insights...")
        weather_context = {
            'location': location_name,
            'current_weather': current_weather,
            'predictions': [p.model_dump() for p in predictions],
            'summary': summary
        }
        
        personalized_insight = generate_gemini_insight(weather_context, user_context)
        
        # Build response
        response = InsightResponse(
            location=location_name,
            latitude=lat,
            longitude=lon,
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            forecast_hours=forecast_hours,
            current_weather=current_weather,
            predictions=predictions,
            summary=summary,
            personalized_insight=personalized_insight,
            profession=user_context.profession,
            model_used=MODEL_TYPE,
            data_source=data_source
        )
        
        logger.info(f"✓ Prediction complete for {location_name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# ============================================================================
# PLANNER ENDPOINTS (for Frontend Planner Page)
# ============================================================================

class PlannerRequest(BaseModel):
    """Request model for AI Planner"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "persona": "driver",
                "location": {"lat": 28.6139, "lon": 77.209, "city": "Delhi"},
                "weatherData": {"current": {"temp": 32, "humidity": 60, "condition": "Clear"}},
                "activity": "travel",
                "date": "2026-02-09",
                "timeRange": {"start": "09:00", "end": "18:00"},
                "risks": ["avoid_rain", "avoid_heat"],
                "duration": 4,
                "notes": "Carrying equipment"
            }
        }
    )
    
    persona: str = Field("general", description="User persona (driver, farmer, worker, etc.)")
    location: Optional[Dict] = Field(None, description="Location with lat, lon, city")
    weatherData: Optional[Dict] = Field(None, description="Current weather data from frontend")
    activity: str = Field(..., description="Activity type (travel, farming, outdoor, event, delivery, exercise, commute, other)")
    date: Optional[str] = Field(None, description="Planned date (YYYY-MM-DD)")
    timeRange: Optional[Dict] = Field(None, description="Time range with start and end times")
    risks: List[str] = Field(default_factory=list, description="Risks to avoid (avoid_rain, avoid_heat, etc.)")
    duration: Optional[int] = Field(4, description="Duration in hours")
    notes: Optional[str] = Field(None, description="Additional notes")

class QuickInsightRequest(BaseModel):
    """Request model for quick dashboard insights"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "weatherData": {"current": {"temp": 32, "humidity": 60, "condition": "Clear"}},
                "persona": "driver",
                "location": {"lat": 28.6139, "lon": 77.209, "city": "Delhi"},
                "weatherRisks": ["heat", "rain"]
            }
        }
    )
    
    weatherData: Dict = Field(..., description="Weather data from frontend")
    persona: str = Field("general", description="User persona")
    location: Optional[Dict] = Field(None, description="Location info")
    weatherRisks: List[str] = Field(default_factory=list, description="User's weather risk preferences")
    timestamp: Optional[str] = Field(None, description="Request timestamp")

@app.post("/quick-insight")
async def get_quick_insight(request: QuickInsightRequest):
    """
    Quick insight endpoint for dashboard - lightweight, fast response
    Uses Gemini for personalized micro-advice
    """
    try:
        logger.info(f"Quick insight request for persona: {request.persona}")
        
        weather = request.weatherData
        current = weather.get('current', {})
        persona = request.persona
        location = request.location
        
        # Try to use Gemini for personalized insight
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                prompt = f"""You are a concise weather advisor for India. Generate a SHORT personalized weather tip.

Current Weather at {location.get('city', 'Unknown') if location else 'Unknown'}:
- Temperature: {current.get('temp', 'N/A')}°C
- Humidity: {current.get('humidity', 'N/A')}%
- Condition: {current.get('condition', 'N/A')}
- Wind: {current.get('wind', 'N/A')} km/h

User Profile:
- Persona: {persona}
- Weather Risks to avoid: {', '.join(request.weatherRisks) if request.weatherRisks else 'None specified'}

Provide a response in this exact JSON format:
{{
  "title": "Short 2-3 word title (e.g., 'Heat Alert', 'Rain Expected', 'Good Conditions')",
  "message": "One concise actionable sentence (max 20 words) tailored to this persona"
}}

Output ONLY the JSON, no other text."""

                response = model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Parse JSON response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    insight = json.loads(json_match.group(0))
                    return {
                        "success": True,
                        "title": insight.get("title", "Today's Tip"),
                        "message": insight.get("message", "Check weather conditions before heading out."),
                        "source": "gemini"
                    }
            except Exception as e:
                logger.warning(f"Gemini quick insight failed: {e}, using fallback")
        
        # Fallback: rule-based insight
        return generate_local_quick_insight(current, persona, request.weatherRisks)
        
    except Exception as e:
        logger.error(f"Quick insight error: {e}")
        return {
            "success": True,
            "title": "Weather Update",
            "message": "Check current conditions before planning outdoor activities.",
            "source": "fallback"
        }

@app.post("/planner")
async def get_smart_plan(request: PlannerRequest):
    """
    Comprehensive AI Planner endpoint - scenario-based predictions
    Uses Gemini for detailed planning advice
    """
    try:
        logger.info(f"Planner request for activity: {request.activity}")
        
        weather = request.weatherData or {}
        current = weather.get('current', {})
        location = request.location or {}
        
        # Try to use Gemini for comprehensive planning
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                prompt = f"""You are an AI weather planner for India. Provide detailed planning advice.

CONTEXT:
Location: {location.get('city', 'Unknown')}
Activity: {request.activity}
Date: {request.date or 'Today'}
Time Range: {request.timeRange.get('start', '09:00')} to {request.timeRange.get('end', '18:00') if request.timeRange else '09:00 to 18:00'}
Duration: {request.duration or 4} hours
User Persona: {request.persona}
Risks to Avoid: {', '.join(request.risks) if request.risks else 'None specified'}
Notes: {request.notes or 'None'}

Current Weather:
- Temperature: {current.get('temp', 'N/A')}°C
- Humidity: {current.get('humidity', 'N/A')}%
- Condition: {current.get('condition', 'N/A')}
- Wind: {current.get('wind', 'N/A')} km/h

Provide response in this exact JSON format:
{{
  "recommendation": "Main recommendation sentence (what to do and when)",
  "bestTime": "Best time window (e.g., '06:00 - 10:00' or '09:00')",
  "avoidTime": "Time to avoid if any (e.g., '12:00 - 16:00' or null)",
  "riskLevel": "Low/Medium/High",
  "tips": ["Tip 1", "Tip 2", "Tip 3", "Tip 4"]
}}

Consider:
- Indian weather patterns
- Activity-specific requirements
- User's persona and risk preferences
- Practical, actionable advice

Output ONLY the JSON, no other text."""

                response = model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Parse JSON response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group(0))
                    return {
                        "success": True,
                        "recommendation": plan.get("recommendation", "Good conditions for your activity."),
                        "bestTime": plan.get("bestTime", request.timeRange.get('start', '09:00') if request.timeRange else '09:00'),
                        "avoidTime": plan.get("avoidTime"),
                        "riskLevel": plan.get("riskLevel", "Low"),
                        "tips": plan.get("tips", ["Check current conditions", "Stay prepared"]),
                        "activity": request.activity,
                        "source": "gemini"
                    }
            except Exception as e:
                logger.warning(f"Gemini planner failed: {e}, using fallback")
        
        # Fallback: rule-based planning
        return generate_local_plan(request)
        
    except Exception as e:
        logger.error(f"Planner error: {e}")
        return {
            "success": True,
            "recommendation": "Unable to generate detailed plan. Check weather conditions before proceeding.",
            "bestTime": "09:00",
            "avoidTime": None,
            "riskLevel": "Medium",
            "tips": ["Check current weather", "Stay prepared for changes"],
            "activity": request.activity,
            "source": "error_fallback"
        }

def generate_local_quick_insight(current: Dict, persona: str, risks: List[str]) -> Dict:
    """Generate rule-based quick insight when Gemini unavailable"""
    temp = current.get('temp', 25)
    humidity = current.get('humidity', 50)
    condition = str(current.get('condition', '')).lower()
    wind = current.get('wind', 0)
    
    title = "Today's Tip"
    message = "Good conditions for your activities today."
    
    if temp >= 35:
        title = "Heat Alert"
        message = "Extreme heat! Stay hydrated and avoid outdoor activities 12-4pm."
    elif temp >= 30:
        message = "High temperature. Carry water and wear light clothes."
    elif temp <= 15:
        title = "Cold Weather"
        message = "Layer up before heading out."
    
    if 'rain' in condition or 'drizzle' in condition:
        title = "Rain Expected"
        message = "Carry umbrella. Roads may be slippery."
    
    if 'fog' in condition or 'mist' in condition:
        title = "Fog Alert"
        message = "Low visibility. Drive carefully."
    
    if wind >= 20:
        title = "Wind Advisory"
        message = "Strong winds expected. Secure loose items."
    
    # Persona-specific adjustments
    if persona in ['driver', 'delivery'] and 'rain' in condition:
        message = "Wet roads ahead. Increase following distance."
    elif persona == 'farmer' and temp >= 32:
        message = "Best farming hours: 5-10 AM. Avoid midday heat."
    elif persona == 'worker' and temp >= 32:
        message = "Take frequent breaks in shade. Hydrate every 30 mins."
    
    return {
        "success": True,
        "title": title,
        "message": message,
        "source": "local"
    }

def generate_local_plan(request: PlannerRequest) -> Dict:
    """Generate rule-based plan when Gemini unavailable"""
    weather = request.weatherData or {}
    current = weather.get('current', {})
    temp = current.get('temp', 25)
    condition = str(current.get('condition', '')).lower()
    
    tips = []
    risk_level = "Low"
    recommendation = "Good conditions for your activity."
    best_time = request.timeRange.get('start', '09:00') if request.timeRange else '09:00'
    avoid_time = None
    
    activity = request.activity
    
    # Activity-specific logic
    if activity in ['travel', 'commute']:
        if 'rain' in condition:
            tips.append("Allow extra travel time due to wet roads")
            tips.append("Check traffic updates before departure")
            risk_level = "Medium"
        if temp >= 35:
            tips.append("Ensure AC is working. Carry water")
            avoid_time = "12:00 - 16:00"
            best_time = "06:00 - 10:00"
    
    elif activity in ['outdoor', 'exercise']:
        if temp >= 32:
            recommendation = "Schedule for early morning (6-9 AM) or evening (5-7 PM)"
            best_time = "06:00 - 09:00"
            avoid_time = "11:00 - 16:00"
            tips.append("Carry water and electrolytes")
            risk_level = "Medium"
        if 'rain' in condition:
            tips.append("Consider indoor alternatives")
            risk_level = "High"
    
    elif activity == 'farming':
        if 'rain' in condition:
            recommendation = "Good day for indoor farm work. Avoid field operations"
            tips.append("Postpone pesticide and fertilizer application")
        if temp >= 35:
            recommendation = "Work in early morning (5-10 AM) or late evening"
            best_time = "05:00 - 10:00"
            avoid_time = "11:00 - 17:00"
            tips.append("Hydrate workers frequently")
    
    elif activity == 'event':
        if 'rain' in condition:
            risk_level = "High"
            recommendation = "Have backup indoor venue ready"
            tips.append("Arrange canopy/tent coverage")
        if temp >= 35:
            tips.append("Arrange shade and cooling stations")
    
    elif activity == 'delivery':
        if 'rain' in condition:
            tips.append("Protect packages from water damage")
            tips.append("Allow extra delivery time")
            risk_level = "Medium"
    
    # Risk adjustments
    if 'avoid_rain' in request.risks and 'rain' in condition:
        risk_level = "High"
        recommendation = "High rain probability. Consider postponing outdoor activities"
    
    if 'avoid_heat' in request.risks and temp >= 32:
        risk_level = "Medium" if risk_level == "Low" else risk_level
        best_time = "06:00 - 10:00"
        avoid_time = "12:00 - 17:00"
    
    if not tips:
        tips = ["Weather conditions are favorable", "Stay aware of any changes in forecast"]
    
    return {
        "success": True,
        "recommendation": recommendation,
        "bestTime": best_time,
        "avoidTime": avoid_time,
        "riskLevel": risk_level,
        "tips": tips,
        "activity": activity,
        "source": "local"
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("🌤️  MAUSAM-VAANI WEATHER API SERVER")
    print("="*80)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Gemini API: {'Configured' if GEMINI_API_KEY else '⚠️  Not configured'}")
    print("="*80)
    print("\nStarting server...")
    print("API Docs: http://localhost:8000/docs")
    print("="*80)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
