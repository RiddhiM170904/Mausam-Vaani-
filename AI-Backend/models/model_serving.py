"""
Model serving utilities for production inference.

This module provides:
- Model loading from checkpoint
- Inference on new data
- Batch prediction
- Real-time prediction for API
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


class WeatherPredictor:
    """Production model serving for weather prediction."""
    
    def __init__(self, checkpoint_path, device='cpu'):
        """
        Initialize predictor with trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Import model here to avoid circular imports
        from models.tft_model import create_model
        
        # Create model from config
        self.model = create_model(checkpoint['config']['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store config
        self.config = checkpoint['config']
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
        print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    
    def preprocess_input(self, historical_data):
        """
        Preprocess historical data for prediction.
        
        Args:
            historical_data: Dictionary or DataFrame with historical weather data
                Required keys/columns: timestamp, temperature, humidity, wind_speed,
                                       rainfall, pressure, cloud_cover, latitude, longitude
        
        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(historical_data, dict):
            df = pd.DataFrame(historical_data)
        else:
            df = historical_data.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add time features (same as in preprocessing)
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Season
        df['season'] = df['month'].apply(self._get_season)
        
        # Select features
        feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'pressure', 'cloud_cover', 'latitude', 'longitude',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'dow_sin', 'dow_cos', 'season'
        ]
        
        features = df[feature_columns].values
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
        
        return features_tensor
    
    @staticmethod
    def _get_season(month):
        """Get season from month (India-specific)."""
        if month in [3, 4, 5]:
            return 0  # Summer
        elif month in [6, 7, 8, 9]:
            return 1  # Monsoon
        elif month in [10, 11]:
            return 2  # Post-monsoon
        else:
            return 3  # Winter
    
    def predict(self, historical_data, forecast_steps=24):
        """
        Make weather prediction.
        
        Args:
            historical_data: Historical weather data (last 168 hours recommended)
            forecast_steps: Number of hours to predict ahead
        
        Returns:
            Dictionary with predictions
        """
        # Preprocess input
        features = self.preprocess_input(historical_data)
        features = features.to(self.device)
        
        # Get encoder steps from feature length
        encoder_steps = features.shape[1]
        
        # Ensure we have enough historical data
        min_encoder_steps = self.config['data'].get('encoder_steps', 168)
        if encoder_steps < min_encoder_steps:
            raise ValueError(
                f"Need at least {min_encoder_steps} hours of historical data, "
                f"but got {encoder_steps}"
            )
        
        # Pad forecast steps to create full input
        # (TFT expects encoder + forecast length, but we only use encoder part)
        padding = torch.zeros(
            1, forecast_steps, features.shape[-1],
            device=self.device
        )
        full_input = torch.cat([features, padding], dim=1)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(
                full_input,
                encoder_steps=encoder_steps,
                forecast_steps=forecast_steps
            )
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Create output dictionary
        target_names = ['temperature', 'humidity', 'wind_speed', 
                       'rainfall', 'pressure', 'cloud_cover']
        
        # Get base timestamp from last historical data point
        if isinstance(historical_data, dict):
            base_timestamp = pd.to_datetime(historical_data['timestamp'][-1])
        else:
            base_timestamp = pd.to_datetime(historical_data['timestamp'].iloc[-1])
        
        # Generate future timestamps
        future_timestamps = [
            base_timestamp + timedelta(hours=i+1) 
            for i in range(forecast_steps)
        ]
        
        # Format predictions
        forecast = []
        for i, timestamp in enumerate(future_timestamps):
            forecast_point = {
                'timestamp': timestamp.isoformat(),
                'hour_ahead': i + 1
            }
            
            for j, name in enumerate(target_names):
                forecast_point[name] = float(predictions[i, j])
            
            forecast.append(forecast_point)
        
        return {
            'base_timestamp': base_timestamp.isoformat(),
            'forecast_steps': forecast_steps,
            'forecast': forecast
        }
    
    def predict_single_point(self, historical_data, hours_ahead=24):
        """
        Predict weather for a specific hour ahead.
        
        Args:
            historical_data: Historical weather data
            hours_ahead: Specific hour to predict (1-24)
        
        Returns:
            Dictionary with prediction for that specific hour
        """
        full_forecast = self.predict(historical_data, forecast_steps=hours_ahead)
        
        # Return only the requested hour
        return full_forecast['forecast'][hours_ahead - 1]
    
    def batch_predict(self, batch_historical_data, forecast_steps=24):
        """
        Make predictions for multiple locations/times.
        
        Args:
            batch_historical_data: List of historical data dictionaries
            forecast_steps: Number of hours to predict
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for historical_data in batch_historical_data:
            try:
                pred = self.predict(historical_data, forecast_steps)
                predictions.append(pred)
            except Exception as e:
                predictions.append({'error': str(e)})
        
        return predictions


def load_predictor(checkpoint_path='checkpoints/best_model.pth', device='cpu'):
    """
    Convenience function to load predictor.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to use
    
    Returns:
        WeatherPredictor instance
    """
    return WeatherPredictor(checkpoint_path, device)


if __name__ == "__main__":
    print("Model serving utilities ready!")
    print("\nUsage example:")
    print("""
    from models.model_serving import load_predictor
    
    # Load model
    predictor = load_predictor('checkpoints/best_model.pth', device='cpu')
    
    # Prepare historical data (last 168 hours)
    historical_data = {
        'timestamp': ['2024-11-23 00:00', '2024-11-23 01:00', ...],  # 168 timestamps
        'temperature': [25.0, 24.8, ...],
        'humidity': [65, 66, ...],
        'wind_speed': [5.2, 5.0, ...],
        'rainfall': [0.0, 0.0, ...],
        'pressure': [1010, 1009, ...],
        'cloud_cover': [20, 25, ...],
        'latitude': [28.6139, 28.6139, ...],  # Same location
        'longitude': [77.2090, 77.2090, ...]
    }
    
    # Make prediction
    forecast = predictor.predict(historical_data, forecast_steps=24)
    
    # Access predictions
    print(f"24-hour forecast from {forecast['base_timestamp']}:")
    for hour_data in forecast['forecast']:
        print(f"  Hour {hour_data['hour_ahead']}: {hour_data['temperature']}°C")
    """)
