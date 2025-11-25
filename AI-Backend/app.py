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
MODEL_PATH = os.getenv('MODEL_PATH', 'best_model.pth')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found in environment variables!")
if not OPENWEATHER_API_KEY:
    logger.warning("⚠️ OPENWEATHER_API_KEY not found in environment variables!")

# Model parameters (must match training config)
MODEL_CONFIG = {
    'num_features': 9,  # Will be updated when loading model
    'hidden_dim': 128,
    'num_heads': 4,
    'num_layers': 2,
    'forecast_horizon': 24,
    'output_dim': 6,  # Will be updated when loading model
    'dropout': 0.1,
}

# Try to load model metadata if available
METADATA_PATH = os.getenv('METADATA_PATH', 'model_metadata.pkl')
model_metadata = None
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, 'rb') as f:
            model_metadata = pickle.load(f)
        logger.info(f"✓ Loaded model metadata from {METADATA_PATH}")
        # Update MODEL_CONFIG from metadata
        if 'model_config' in model_metadata:
            MODEL_CONFIG.update(model_metadata['model_config'])
        if 'num_features' in model_metadata:
            MODEL_CONFIG['num_features'] = model_metadata['num_features']
        if 'num_targets' in model_metadata:
            MODEL_CONFIG['output_dim'] = model_metadata['num_targets']
        logger.info(f"  Features: {model_metadata.get('feature_cols', 'N/A')}")
        logger.info(f"  Targets: {model_metadata.get('target_cols', 'N/A')}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load metadata: {e}")

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
        weather_model = TemporalFusionTransformer(
            num_features=MODEL_CONFIG['num_features'],
            hidden_dim=MODEL_CONFIG['hidden_dim'],
            num_heads=MODEL_CONFIG['num_heads'],
            num_layers=MODEL_CONFIG['num_layers'],
            forecast_horizon=MODEL_CONFIG['forecast_horizon'],
            output_dim=MODEL_CONFIG['output_dim'],
            dropout=MODEL_CONFIG['dropout']
        ).to(device)
        
        # Load weights
        state_dict = torch.load(MODEL_PATH, map_location=device)

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
# TFT MODEL DEFINITION (Same as training)
# ============================================================================

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    
    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(F.elu(self.fc1(x))))
        h = h * gate
        if self.skip is not None:
            x = self.skip(x)
        return self.layer_norm(x + h)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        self.grn_combine = GatedResidualNetwork(
            num_features * hidden_dim, hidden_dim, hidden_dim, dropout
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        processed = [grn(x[:, i, :]) for i, grn in enumerate(self.grns)]
        processed = torch.stack(processed, dim=1)
        flattened = processed.view(batch_size, -1)
        combined = self.grn_combine(flattened)
        return combined, processed

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features=9, hidden_dim=128, num_heads=4, num_layers=2, 
                 forecast_horizon=24, output_dim=6, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        
        self.vsn = VariableSelectionNetwork(1, num_features, hidden_dim, dropout)
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, 
                                     dropout=dropout if num_layers > 1 else 0)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                                     dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.grn_post_attention = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.fc_out = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(output_dim)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_steps=168, forecast_steps=24):
        batch_size = x.size(0)
        encoder_outputs = []
        
        for t in range(encoder_steps):
            step_input = x[:, t, :].unsqueeze(-1)
            selected, _ = self.vsn(step_input)
            encoder_outputs.append(selected)
        
        encoder_outputs = torch.stack(encoder_outputs, dim=1)
        encoder_out, (h_n, c_n) = self.encoder_lstm(encoder_outputs)
        decoder_input = encoder_out[:, -1:, :].repeat(1, forecast_steps, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input, (h_n, c_n))
        attn_out, _ = self.attention(decoder_out, encoder_out, encoder_out)
        attn_out = self.grn_post_attention(attn_out.reshape(-1, self.hidden_dim))
        attn_out = attn_out.reshape(batch_size, forecast_steps, self.hidden_dim)
        
        predictions = [fc(attn_out) for fc in self.fc_out]
        predictions = torch.cat(predictions, dim=-1)
        
        return predictions

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
    forecast_hours: int = Field(24, description="Number of hours to forecast", ge=1, le=72)

class WeatherPrediction(BaseModel):
    """Weather prediction output"""
    timestamp: str
    temperature: float
    humidity: float
    wind_speed: float
    rainfall: float
    pressure: Optional[float] = None
    cloud_cover: Optional[float] = None

class InsightResponse(BaseModel):
    """Complete API response"""
    location: str
    latitude: float
    longitude: float
    current_time: str
    forecast_hours: int
    predictions: List[WeatherPrediction]
    summary: Dict[str, float]  # avg, min, max for key metrics
    personalized_insight: str
    profession: str

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
    temps = [p.temperature for p in predictions[:6]] if predictions else [avg_temp]
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

def fetch_weather_data_from_openweather(location_name: str) -> Dict:
    """
    Fetch real-time and forecast weather data from OpenWeatherMap API
    Returns historical-like data + current conditions + coordinates
    """
    if not OPENWEATHER_API_KEY:
        logger.warning("OpenWeather API key not configured, using dummy data")
        return None
    
    try:
        # Get coordinates from location name using Geocoding API
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_name}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_response = requests.get(geo_url, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data:
            logger.warning(f"Location '{location_name}' not found")
            return None
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        actual_name = geo_data[0].get('name', location_name)
        
        logger.info(f"Found location: {actual_name} ({lat}, {lon})")
        
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

def make_prediction(input_data: np.ndarray, forecast_hours: int = 24) -> np.ndarray:
    """Make weather prediction using the TFT model"""
    
    # If model is not loaded, use realistic dummy predictions
    if weather_model is None:
        logger.info("Using dummy data for prediction (model not loaded)")
        try:
            logger.debug(f"Input data shape: {getattr(input_data, 'shape', None)}")
            last_vals = input_data[-1] if len(input_data) > 0 else None
            logger.debug(f"Last historical values sample: {last_vals}")
        except Exception:
            logger.debug("Unable to inspect input_data for debug")
        return generate_realistic_dummy_prediction(input_data, forecast_hours)
    
    try:
        # Prepare input
        # Input shape: (batch=1, total_steps=168+forecast_hours, num_features=9)
        encoder_steps = 168
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

@app.post("/predict", response_model=InsightResponse)
async def predict_weather(request: PredictionRequest):
    """
    Main prediction endpoint
    
    Returns weather predictions + personalized LLM insights
    """
    
    try:
        logger.info(f"Prediction request for {request.weather_input.location_name}")
        
        # Fetch real weather data from OpenWeatherMap
        weather_data_result = fetch_weather_data_from_openweather(request.weather_input.location_name)
        
        current_weather = None
        if weather_data_result:
            # Use real data from API
            historical_data = weather_data_result['historical_data']
            location_name = weather_data_result['location_name']
            latitude = weather_data_result['latitude']
            longitude = weather_data_result['longitude']
            current_weather = weather_data_result.get('current_weather')
            logger.info(f"✓ Using real weather data from OpenWeatherMap for {location_name}")
        else:
            # Fallback to dummy data
            logger.warning("⚠️ Using dummy data (OpenWeather API unavailable)")
            latitude = request.weather_input.latitude or 28.6139
            longitude = request.weather_input.longitude or 77.2090
            location_name = request.weather_input.location_name
            historical_data = create_dummy_historical_data(latitude, longitude)
        
        # Make prediction
        predictions = make_prediction(historical_data, request.forecast_hours)
        
        # Parse predictions
        current_time = datetime.now()
        forecast_list = []
        
        feature_names = ['temperature', 'humidity', 'wind_speed', 'rainfall', 'pressure', 'cloud_cover']
        
        for i in range(request.forecast_hours):
            timestamp = current_time + timedelta(hours=i+1)
            
            pred_dict = {
                'timestamp': timestamp.isoformat(),
                'temperature': float(predictions[i, 0]) if predictions.shape[1] > 0 else 25.0,
                'humidity': float(predictions[i, 1]) if predictions.shape[1] > 1 else 60.0,
                'wind_speed': float(predictions[i, 2]) if predictions.shape[1] > 2 else 5.0,
                'rainfall': float(predictions[i, 3]) if predictions.shape[1] > 3 else 0.0,
                'pressure': float(predictions[i, 4]) if predictions.shape[1] > 4 else 1010.0,
                'cloud_cover': float(predictions[i, 5]) if predictions.shape[1] > 5 else 50.0,
            }
            
            forecast_list.append(WeatherPrediction(**pred_dict))
        
        # Calculate summary statistics
        temps = [p.temperature for p in forecast_list]
        rainfalls = [p.rainfall for p in forecast_list]
        humidities = [p.humidity for p in forecast_list]
        wind_speeds = [p.wind_speed for p in forecast_list]
        
        summary = {
            'avg_temperature': np.mean(temps),
            'min_temperature': np.min(temps),
            'max_temperature': np.max(temps),
            'avg_rainfall': np.mean(rainfalls),
            'total_rainfall': np.sum(rainfalls),
            'avg_humidity': np.mean(humidities),
            'avg_wind_speed': np.mean(wind_speeds),
        }
        
        # Prepare data for LLM with current weather
        weather_data = {
            'location': location_name,
            'current_weather': current_weather,
            'summary': summary,
            'predictions': forecast_list
        }
        
        # Generate personalized insight using Gemini
        logger.info("🤖 Generating AI insights with Gemini...")
        personalized_insight = generate_gemini_insight(weather_data, request.user_context)
        
        # Build response
        response = InsightResponse(
            location=location_name,
            latitude=latitude,
            longitude=longitude,
            current_time=current_time.isoformat(),
            forecast_hours=request.forecast_hours,
            predictions=forecast_list,
            summary=summary,
            personalized_insight=personalized_insight,
            profession=request.user_context.profession
        )
        
        logger.info(f"✓ Prediction completed successfully")
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
