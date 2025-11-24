"""
API routes for weather prediction and insights.

Endpoints:
- POST /api/predict-weather: Get weather predictions
- POST /api/get-insight: Get personalized insights with Gemini
"""

from flask import Blueprint, request, jsonify, current_app
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.model_serving import load_predictor
from gemini_integration import generate_personalized_insight
from utils import validate_historical_data, validate_insight_request

logger = logging.getLogger(__name__)

# Create blueprint
weather_bp = Blueprint('weather', __name__)

# Global predictor (loaded once)
_predictor = None


def get_predictor():
    """Get or initialize the model predictor."""
    global _predictor
    
    if _predictor is None:
        model_path = current_app.config.get('MODEL_PATH', 'checkpoints/best_model.pth')
        device = current_app.config.get('DEVICE', 'cpu')
        
        logger.info(f"Loading model from: {model_path}")
        
        try:
            _predictor = load_predictor(model_path, device=device)
            current_app.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    return _predictor


@weather_bp.route('/predict-weather', methods=['POST'])
def predict_weather():
    """
    Predict weather based on historical data.
    
    Request Body:
    {
        "historical_data": {
            "timestamp": ["2024-11-23 00:00", ...],  // 168 timestamps
            "temperature": [25.0, ...],
            "humidity": [65, ...],
            "wind_speed": [5.2, ...],
            "rainfall": [0.0, ...],
            "pressure": [1010, ...],
            "cloud_cover": [20, ...],
            "latitude": [28.6139, ...],
            "longitude": [77.2090, ...]
        },
        "forecast_steps": 24  // Optional, default 24
    }
    
    Response:
    {
        "success": true,
        "base_timestamp": "2024-11-24T00:00:00",
        "forecast_steps": 24,
        "forecast": [
            {
                "timestamp": "2024-11-24T01:00:00",
                "hour_ahead": 1,
                "temperature": 25.3,
                "humidity": 66,
                "wind_speed": 5.1,
                "rainfall": 0.0,
                "pressure": 1009,
                "cloud_cover": 22
            },
            ...
        ]
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate input
        historical_data = data.get('historical_data')
        if not historical_data:
            return jsonify({
                'success': False,
                'error': 'historical_data is required'
            }), 400
        
        # Validate historical data format
        validation_error = validate_historical_data(historical_data)
        if validation_error:
            return jsonify({
                'success': False,
                'error': validation_error
            }), 400
        
        forecast_steps = data.get('forecast_steps', 24)
        
        # Validate forecast steps
        if not isinstance(forecast_steps, int) or forecast_steps < 1 or forecast_steps > 168:
            return jsonify({
                'success': False,
                'error': 'forecast_steps must be an integer between 1 and 168'
            }), 400
        
        # Get predictor
        predictor = get_predictor()
        
        # Make prediction
        logger.info(f"Making prediction for {forecast_steps} hours ahead")
        result = predictor.predict(historical_data, forecast_steps=forecast_steps)
        
        # Return response
        return jsonify({
            'success': True,
            **result
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate prediction'
        }), 500


@weather_bp.route('/get-insight', methods=['POST'])
def get_insight():
    """
    Get personalized weather insight using Gemini.
    
    Request Body:
    {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "city": "Delhi",  // Optional
        "user_profession": "Farmer",  // e.g., Farmer, Commuter, Construction Worker
        "user_context": {  // Optional additional context
            "crop": "Rice",
            "activity": "Outdoor sports",
            "transport": "Two-wheeler"
        },
        "historical_data": {...},  // Same as predict-weather
        "forecast_steps": 24  // Optional
    }
    
    Response:
    {
        "success": true,
        "location": {
            "latitude": 28.6139,
            "longitude": 77.2090,
            "city": "Delhi"
        },
        "weather": {
            "current": {...},
            "forecast_summary": "Heavy rain expected in next 6 hours"
        },
        "personalized_insight": "Due to heavy rainfall, avoid pesticide spraying today..."
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate input
        validation_error = validate_insight_request(data)
        if validation_error:
            return jsonify({
                'success': False,
                'error': validation_error
            }), 400
        
        # Get weather prediction first
        historical_data = data.get('historical_data')
        forecast_steps = data.get('forecast_steps', 24)
        
        predictor = get_predictor()
        weather_forecast = predictor.predict(historical_data, forecast_steps=forecast_steps)
        
        # Extract current and future weather
        latest_forecast = weather_forecast['forecast'][0]  # 1 hour ahead
        
        # Prepare weather context for Gemini
        weather_context = {
            'temperature': latest_forecast['temperature'],
            'humidity': latest_forecast['humidity'],
            'wind_speed': latest_forecast['wind_speed'],
            'rainfall': latest_forecast['rainfall'],
            'pressure': latest_forecast['pressure'],
            'cloud_cover': latest_forecast['cloud_cover']
        }
        
        # Determine weather condition
        if latest_forecast['rainfall'] > 5:
            condition = "Heavy Rain"
        elif latest_forecast['rainfall'] > 0:
            condition = "Light Rain"
        elif latest_forecast['cloud_cover'] > 70:
            condition = "Cloudy"
        elif latest_forecast['temperature'] > 35:
            condition = "Very Hot"
        else:
            condition = "Clear"
        
        # Get Gemini API key
        gemini_api_key = current_app.config.get('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.warning("Gemini API key not configured")
            return jsonify({
                'success': False,
                'error': 'Gemini API key not configured'
            }), 500
        
        # Generate personalized insight
        logger.info(f"Generating insight for {data.get('user_profession', 'Unknown')} user")
        
        insight_params = {
            'location': data.get('city', f"Lat {data['latitude']}, Lon {data['longitude']}"),
            'condition': condition,
            'temperature': weather_context['temperature'],
            'humidity': weather_context['humidity'],
            'rainfall': weather_context['rainfall'],
            'wind_speed': weather_context['wind_speed'],
            'user_profession': data.get('user_profession', 'General'),
            'user_context': data.get('user_context', {})
        }
        
        personalized_insight = generate_personalized_insight(
            insight_params,
            api_key=gemini_api_key
        )
        
        # Prepare response
        response = {
            'success': True,
            'location': {
                'latitude': data['latitude'],
                'longitude': data['longitude'],
                'city': data.get('city', 'Unknown')
            },
            'weather': {
                'current': weather_context,
                'condition': condition,
                'forecast_summary': f"{condition} expected"
            },
            'personalized_insight': personalized_insight
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate insight'
        }), 500


@weather_bp.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to check if API is working."""
    return jsonify({
        'success': True,
        'message': 'Mausam-Vaani API is running'
    }), 200
