"""
Flask API for Mausam-Vaani Weather Prediction

This is the main Flask application that serves:
- Weather prediction endpoint
- Personalized insights with Gemini integration
- Health check endpoint
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import logging

# Import routes
from routes import weather_bp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config=None):
    """
    Create and configure Flask application.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    
    # Load configuration
    if config is None:
        config = {
            'MODEL_PATH': os.getenv('MODEL_PATH', 'checkpoints/best_model.pth'),
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', ''),
            'DEVICE': os.getenv('DEVICE', 'cpu'),
            'DEBUG': os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
            'PORT': int(os.getenv('PORT', 5000)),
            'HOST': os.getenv('HOST', '0.0.0.0')
        }
    
    app.config.update(config)
    
    # Enable CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:5173", "*"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Register blueprints
    app.register_blueprint(weather_bp, url_prefix='/api')
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model_loaded': hasattr(app, 'model_loaded') and app.model_loaded,
            'gemini_configured': bool(app.config.get('GEMINI_API_KEY'))
        }), 200
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with API information."""
        return jsonify({
            'name': 'Mausam-Vaani AI Backend',
            'version': '1.0.0',
            'description': 'Hyperlocal weather prediction with AI',
            'endpoints': {
                'health': '/health',
                'predict_weather': '/api/predict-weather',
                'get_insight': '/api/get-insight'
            }
        }), 200
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    logger.info("Flask app created successfully")
    
    return app


if __name__ == "__main__":
    # Create app
    app = create_app()
    
    # Get config
    host = app.config.get('HOST', '0.0.0.0')
    port = app.config.get('PORT', 5000)
    debug = app.config.get('DEBUG', False)
    
    logger.info(f"Starting Mausam-Vaani AI Backend on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Run app
    app.run(host=host, port=port, debug=debug)
