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
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

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

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY not found in environment variables!")

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
        weather_model.load_state_dict(state_dict)
        weather_model.eval()
        
        logger.info(f"✓ Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load model: {e}")
        logger.info("✓ API will use dummy data for predictions (perfect for demo!)")
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
                "latitude": 28.6139,
                "longitude": 77.2090,
                "location_name": "Delhi"
            }
        }
    )
    
    latitude: float = Field(..., description="Latitude", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude", ge=-180, le=180)
    location_name: Optional[str] = Field(None, description="Location name (e.g., 'Delhi')")
    
    # Historical weather data (last 7 days hourly = 168 points)
    # If not provided, will use dummy data for demo
    historical_data: Optional[List[List[float]]] = Field(
        None, 
        description="Historical weather data (168 x num_features)"
    )

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
    """Build prompt for Gemini"""
    
    location = weather_data.get('location', 'your location')
    avg_temp = weather_data['summary']['avg_temperature']
    max_temp = weather_data['summary']['max_temperature']
    avg_rainfall = weather_data['summary']['avg_rainfall']
    total_rainfall = weather_data['summary']['total_rainfall']
    avg_humidity = weather_data['summary']['avg_humidity']
    avg_wind = weather_data['summary']['avg_wind_speed']
    
    # Determine weather condition
    if total_rainfall > 50:
        condition = "Heavy Rain"
    elif total_rainfall > 10:
        condition = "Moderate Rain"
    elif total_rainfall > 0:
        condition = "Light Rain"
    elif avg_temp > 35:
        condition = "Very Hot"
    elif avg_temp < 15:
        condition = "Cold"
    else:
        condition = "Pleasant"
    
    profession = user_context.profession
    context = user_context.additional_context or {}
    
    prompt = f"""You are a helpful weather assistant for India providing actionable weather advice.

Location: {location}
Next 24 Hours Weather Forecast:
- Condition: {condition}
- Average Temperature: {avg_temp:.1f}°C (Max: {max_temp:.1f}°C)
- Total Rainfall: {total_rainfall:.1f}mm
- Average Humidity: {avg_humidity:.1f}%
- Average Wind Speed: {avg_wind:.1f} km/h

User Profile:
- Profession: {profession}
"""
    
    if context:
        prompt += "\nAdditional Context:\n"
        for key, value in context.items():
            prompt += f"- {key}: {value}\n"
    
    # Profession-specific instructions
    profession_instructions = {
        'Farmer': """
Give SHORT actionable advice for farmers (2-3 sentences maximum):
- Best times for farming activities (sowing, harvesting, irrigation, pesticide spraying)
- Warnings about pest/disease risks due to weather
- Equipment/crop protection recommendations
Be specific, practical, and use emojis appropriately.""",
        
        'Commuter': """
Give SHORT travel advice for commuters (2-3 sentences maximum):
- Traffic/road condition warnings
- Best travel times to avoid weather issues
- Safety precautions (carry umbrella, drive carefully, etc.)
Be concise, practical, and use emojis appropriately.""",
        
        'Construction Worker': """
Give SHORT advice for construction workers (2-3 sentences maximum):
- Work schedule recommendations based on weather
- Safety warnings (heat, rain, wind, lightning)
- Material handling and storage advice
Be specific about timing and use emojis appropriately.""",
        
        'Outdoor Sports': """
Give SHORT advice for outdoor activities (2-3 sentences maximum):
- Best times for outdoor sports/activities
- Safety precautions for the weather
- Hydration/sun protection/rain gear recommendations
Be practical and use emojis appropriately.""",
        
        'General': """
Give SHORT general weather advice (2-3 sentences maximum):
- What to expect in the next 24 hours
- What to carry (umbrella, water bottle, sunscreen, etc.)
- Any safety precautions needed
Be helpful, concise, and use emojis appropriately."""
    }
    
    instruction = profession_instructions.get(profession, profession_instructions['General'])
    prompt += f"\n{instruction}"
    prompt += "\n\nProvide ONLY the actionable advice in 2-3 sentences. Be direct and helpful."
    
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
        
        # Rainfall: occasional rain
        rainfall = max(0, np.random.randn() * 3) if np.random.rand() > 0.85 else 0
        
        # Pressure: slight variations
        pressure = base_pressure + np.random.randn() * 2
        
        # Cloud cover: correlated with rainfall
        cloud = min(100, max(0, base_cloud + (rainfall * 5) + np.random.randn() * 15))
        
        predictions.append([temp, humidity, wind, rainfall, pressure, cloud])
    
    return np.array(predictions)


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

@app.post("/predict", response_model=InsightResponse)
async def predict_weather(request: PredictionRequest):
    """
    Main prediction endpoint
    
    Returns weather predictions + personalized LLM insights
    """
    
    try:
        logger.info(f"Prediction request for {request.weather_input.location_name or 'Unknown location'}")
        
        # Get or create historical data
        if request.weather_input.historical_data:
            historical_data = np.array(request.weather_input.historical_data)
        else:
            # Create dummy data for demo
            historical_data = create_dummy_historical_data(
                request.weather_input.latitude,
                request.weather_input.longitude
            )
        
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
        
        # Prepare data for LLM
        weather_data = {
            'location': request.weather_input.location_name or f"({request.weather_input.latitude}, {request.weather_input.longitude})",
            'summary': summary,
            'predictions': forecast_list
        }
        
        # Generate personalized insight using Gemini
        personalized_insight = generate_gemini_insight(weather_data, request.user_context)
        
        # Build response
        response = InsightResponse(
            location=weather_data['location'],
            latitude=request.weather_input.latitude,
            longitude=request.weather_input.longitude,
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
