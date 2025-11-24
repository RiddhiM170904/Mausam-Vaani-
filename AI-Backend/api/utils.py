"""
Utility functions for API validation and helpers.
"""

import logging

logger = logging.getLogger(__name__)


def validate_historical_data(data):
    """
    Validate historical weather data format.
    
    Args:
        data: Historical data dictionary or DataFrame
    
    Returns:
        Error message if invalid, None if valid
    """
    required_fields = [
        'timestamp', 'temperature', 'humidity', 'wind_speed',
        'rainfall', 'pressure', 'cloud_cover', 'latitude', 'longitude'
    ]
    
    # Check if data is dict or dict-like
    if not isinstance(data, dict):
        return "historical_data must be a dictionary"
    
    # Check for required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return f"Missing required fields: {', '.join(missing_fields)}"
    
    # Check if all fields have same length
    lengths = {field: len(data[field]) for field in required_fields}
    unique_lengths = set(lengths.values())
    
    if len(unique_lengths) > 1:
        return f"All fields must have same length. Got: {lengths}"
    
    # Check minimum length (need at least 24 hours, ideally 168)
    data_length = list(unique_lengths)[0]
    if data_length < 24:
        return f"Need at least 24 hours of historical data, got {data_length}"
    
    # Validate data types
    numeric_fields = [
        'temperature', 'humidity', 'wind_speed',
        'rainfall', 'pressure', 'cloud_cover', 'latitude', 'longitude'
    ]
    
    for field in numeric_fields:
        try:
            # Try to convert to float
            values = [float(v) for v in data[field]]
        except (ValueError, TypeError):
            return f"Field '{field}' must contain numeric values"
    
    # Validate value ranges
    if any(h < 0 or h > 100 for h in data['humidity']):
        return "Humidity must be between 0 and 100"
    
    if any(cc < 0 or cc > 100 for cc in data['cloud_cover']):
        return "Cloud cover must be between 0 and 100"
    
    if any(r < 0 for r in data['rainfall']):
        return "Rainfall cannot be negative"
    
    if any(ws < 0 for ws in data['wind_speed']):
        return "Wind speed cannot be negative"
    
    # Validate lat/lon ranges
    if any(lat < -90 or lat > 90 for lat in data['latitude']):
        return "Latitude must be between -90 and 90"
    
    if any(lon < -180 or lon > 180 for lon in data['longitude']):
        return "Longitude must be between -180 and 180"
    
    return None  # All validations passed


def validate_insight_request(data):
    """
    Validate insight request data.
    
    Args:
        data: Request data dictionary
    
    Returns:
        Error message if invalid, None if valid
    """
    # Check required fields
    if 'latitude' not in data:
        return "latitude is required"
    
    if 'longitude' not in data:
        return "longitude is required"
    
    if 'historical_data' not in data:
        return "historical_data is required"
    
    # Validate latitude and longitude
    try:
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        
        if lat < -90 or lat > 90:
            return "latitude must be between -90 and 90"
        
        if lon < -180 or lon > 180:
            return "longitude must be between -180 and 180"
    except (ValueError, TypeError):
        return "latitude and longitude must be numeric"
    
    # Validate historical data
    historical_error = validate_historical_data(data['historical_data'])
    if historical_error:
        return f"historical_data validation failed: {historical_error}"
    
    # Validate optional fields
    if 'user_profession' in data:
        valid_professions = [
            'Farmer', 'Commuter', 'Construction Worker',
            'Outdoor Sports', 'General', 'Business', 'Tourism'
        ]
        
        if data['user_profession'] not in valid_professions:
            logger.warning(f"Unknown profession: {data['user_profession']}")
            # Don't reject, just log warning
    
    if 'user_context' in data:
        if not isinstance(data['user_context'], dict):
            return "user_context must be a dictionary"
    
    return None  # All validations passed


def format_error_response(error_msg):
    """
    Format error response consistently.
    
    Args:
        error_msg: Error message
    
    Returns:
        Dictionary for JSON response
    """
    return {
        'success': False,
        'error': error_msg
    }


def format_success_response(data):
    """
    Format success response consistently.
    
    Args:
        data: Response data
    
    Returns:
        Dictionary for JSON response
    """
    return {
        'success': True,
        **data
    }


if __name__ == "__main__":
    print("API utilities module ready!")
