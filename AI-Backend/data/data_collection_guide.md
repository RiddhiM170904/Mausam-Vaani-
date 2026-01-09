# Data Collection Guide for Mausam-Vaani

## Overview
This guide explains how to collect and prepare weather data for training the TFT model.

## Required Data Format

The model requires **hourly weather data** in CSV format with the following columns:

### Weather & Location Columns

| Column | Type | Description | Unit | Example |
|--------|------|-------------|------|---------|
| `timestamp` | datetime | Date and time | YYYY-MM-DD HH:MM | 2024-07-01 14:00 |
| `city` | string | City name | - | Delhi |
| `latitude` | float | Latitude coordinate | degrees | 28.6139 |
| `longitude` | float | Longitude coordinate | degrees | 77.2090 |
| `temperature` | float | Air temperature | °C | 32.5 |
| `humidity` | float | Relative humidity | % (0-100) | 65 |
| `wind_speed` | float | Wind speed | km/h | 2.5 |
| `rainfall` | float | Rainfall amount | mm | 0.0 |
| `pressure` | float | Atmospheric pressure | hPa | 1005 |
| `cloud_cover` | float | Cloud coverage | % (0-100) | 20 |

### Air Quality Column

| Column | Type | Description | Unit | Example |
|--------|------|-------------|------|---------|
| `aqi` | integer | Air Quality Index | AQI scale (0-500) | 152 |
| `pm25` | float | PM2.5 concentration | µg/m³ | 45.2 |
| `pm10` | float | PM10 concentration | µg/m³ | 85.7 |
| `co` | float | Carbon monoxide | µg/m³ | 350.0 |
| `no2` | float | Nitrogen dioxide | µg/m³ | 25.5 |
| `o3` | float | Ozone | µg/m³ | 60.2 |
| `so2` | float | Sulfur dioxide | µg/m³ | 15.8 |

### Astronomical Columns

| Column | Type | Description | Unit | Example |
|--------|------|-------------|------|---------|
| `sunrise` | time | Sunrise time | HH:MM | 05:45 |
| `sunset` | time | Sunset time | HH:MM | 19:15 |
| `moonrise` | time | Moonrise time | HH:MM | 22:30 |
| `moonset` | time | Moonset time | HH:MM | 11:20 |
| `moon_phase` | float | Moon phase | 0-1 (0=new, 0.5=full) | 0.75 |
| `day_length` | float | Daylight duration | hours | 13.5 |

## Complete CSV File Structure

Your combined CSV file should include ALL the following columns:

```csv
timestamp,city,latitude,longitude,temperature,humidity,wind_speed,rainfall,pressure,cloud_cover,aqi,pm25,pm10,co,no2,o3,so2,sunrise,sunset,moonrise,moonset,moon_phase,day_length
2024-07-01 00:00,Delhi,28.6139,77.2090,30.2,72,8.5,0.0,1008,35,168,52.3,95.6,400.0,28.3,65.8,18.2,05:45,19:15,22:30,11:20,0.75,13.5
2024-07-01 01:00,Delhi,28.6139,77.2090,29.8,74,7.2,0.0,1008,40,165,50.1,92.3,385.0,27.1,63.2,17.5,05:45,19:15,22:30,11:20,0.75,13.5
2024-07-01 02:00,Delhi,28.6139,77.2090,29.5,75,6.8,0.0,1009,45,162,48.5,89.7,370.0,26.0,61.5,16.8,05:45,19:15,22:30,11:20,0.75,13.5
```

## Minimum Data Requirements

- **Duration**: At least **6 months** of continuous data (ideally 1-2 years)
- **Frequency**: **Hourly** measurements (one row per hour)
- **Cities**: At least **5-10 major Indian cities** for better generalization
- **Total Records**: Minimum ~43,800 rows (6 months × 730 hours/month × 10 cities)
- **Columns**: All 23 columns listed above (10 weather + 7 air quality + 6 astronomical)

## Data Sources

### 1. India Meteorological Department (IMD)
**Website**: https://mausam.imd.gov.in/

**Access**:
- Historical data archives
- Real-time API (requires registration)
- State weather centers

**How to get**:
1. Visit IMD website
2. Navigate to "Data/Services" → "Historical Data"
3. Request data for specific stations
4. May require official request/payment for bulk data

### 2. OpenWeatherMap (Weather + Air Quality)
**Website**: https://openweathermap.org/

**Access**:
- Free tier: 1,000 calls/day
- Weather API: https://openweathermap.org/api
- Air Quality API: https://openweathermap.org/api/air-pollution
- Astronomy data included in One Call API

**How to use**:
```python
import requests

API_KEY = "your_api_key"
lat, lon = 28.6139, 77.2090  # Delhi

# Weather data
weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
weather_data = requests.get(weather_url).json()

# Air Quality data (Comprehensive)
**Website**: https://www.visualcrossing.com/

**Access**:
- Free tier: 1,000 records/day
- Historical weather data API
- **Includes**: Weather, AQI, sunrise/sunset, moon data
- Good for Indian cities

**How to use**:
1. Sign up at https://www.visualcrossing.com/sign-up
2. Get API key
3. Use Timeline Weather API with all elements
4. Export as CSV

**Example API call** (includes all required fields):
```
https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Delhi/2024-01-01/2024-06-30?unitGroup=metric&include=hours&elements=datetime,temp,humidity,windspeed,precip,pressure,cloudcover,sunrise,sunset,moonphase,moonrise,moonset,visibility,solarradiation&key=YOUR_API_KEY
```

**Note**: For AQI data, you may need to combine with another source like OpenWeatherMap or IQAir.ow to use**:
1. Sign up at https://www.visualcrossing.com/sign-up
2. Get API key
3. Use Timeline Weather API
4. Export as CSV

**Example API call**:
```
https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Delhi/2024-01-01/2024-06-30?unitGroup=metric&key=YOUR_API_KEY
```

### 4. Government Open Data Portal
**Website**: https://data.gov.in/

**How to search**:
1. Search for "weather" or "meteorological"
2. FiltIQAir (Air Quality Specialist)
**Website**: https://www.iqair.com/air-pollution-data-api

**Features**:
- Comprehensive AQI data (PM2.5, PM10, CO, NO2, O3, SO2)
- Real-time and historical data
- Free tier available
- Excellent for Indian cities

**How to use**:
```python
import requests
Complete Data Collection Script

```python
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# API Keys
OPENWEATHER_API_KEY = "your_openweather_key"
IQAIR_API_KEY = "your_iqair_key"

cities = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi", "country": "India"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "country": "India"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "country": "India"}
}

def collect_comprehensive_data(city, lat, lon, timestamp):
    """Collect all required data for one hour"""
    
    # 1. Weather Data (OpenWeatherMap)
    weather_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={int(timestamp.timestamp())}&appid={OPENWEATHER_API_KEY}"
    weather_data = requests.get(weather_url).json()
    
    # 2. Air Quality Data (OpenWeatherMap Air Pollution API)
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    aqi_data = requests.get(aqi_url).json()
    
    # Extract and combine data
    row = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        "city": city,
        "latitude": lat,
        "longitude": lon,
        # Weather
        "temperature": weather_data['current']['temp'] - 273.15,  # Convert to Celsius
        "humidity": weather_data['current']['humidity'],
        "wind_speed": weather_data['current']['wind_speed'] * 3.6,  # m/s to km/h
        "rainfall": weather_data['current'].get('rain', {}).get('1h', 0),
        "pressure": weather_data['current']['pressure'],
        "cloud_cover": weather_data['current']['clouds'],
        # Air Quality
        "aqi": aqi_data['list'][0]['main']['aqi'] * 50,  # Convert to US AQI scale
        "pm25": aqi_data['list'][0]['components']['pm2_5'],
        "pm10": aqi_data['list'][0]['components']['pm10'],
        "co": aqi_data['list'][0]['components']['co'],
        "no2": aqi_data['list'][0]['components']['no2'],
        "o3": aqi_data['list'][0]['components']['o3'],
        "so2": aqi_data['list'][0]['components']['so2'],
        # Astronomical
        "sunrise": datetime.fromtimestamp(weather_data['current']['sunrise']).strftime("%H:%M"),
        "sunset": datetime.fromtimestamp(weather_data['current']['sunset']).strftime("%H:%M"),
        "moonrise": datetime.fromtimestamp(weather_data['current'].get('moonrise', 0)).strftime("%H:%M") if 'moonrise' in weather_data['current'] else None,
        "moonset": datetime.fromtimestamp(weather_data['current'].get('moonset', 0)).strftime("%H:%M") if 'moonset' in weather_data['current'] else None,
        "moon_phase": weather_data['current'].get('moon_phase', 0),
        "day_length": (weather_data['current']['sunset'] - weather_data['current']['sunrise']) / 3600
    }
    
    return row

# Collect data
all_data = []
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 6, 30)

for city, coords in cities.items():
    current_date = start_date
    while current_date <= end_date:
        for hour in range(24):
            timestamp = current_date + timedelta(hours=hour)
            try:
                row = collect_comprehensive_data(city, coords['lat'], coords['lon'], timestamp)
                all_data.append(row)
                print(f"Collected: {city} - {timestamp}")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error for {city} at {timestamp}: {e}")
        
        current_date += timedelta(days=1)

# Save to CSV
df = pd.DataFrame(all_data)
df.to_csv('comprehensive_weather_data.csv', index=False)
print(f"Saved {len(df)} records to comprehensive_weather_data.csv")ervice but reasonable rates

## Data Collection Script Example

We've provided scripts to help you collect data:

### Using OpenWeatherMap (see `scripts/download_imd_data.py`)
 or forward-fill
2. **Consistent timestamps**: Hourly intervals without gaps
3. **Valid ranges**:
   - **Weather**:
     - Temperature: -10°C to 50°C (India)
     - Humidity: 0-100%
     - Wind speed: ≥ 0 km/h
     - Rainfall: ≥ 0 mm
     - Pressure: 900-1100 hPa
     - Cloud cover: 0-100%
   - **Air Quality**:
     - AQI: 0-500
     - PM2.5: 0-500 µg/m³
     - PM10: 0-600 µg/m³
     - CO: ≥ 0 µg/m³
     - NO2: ≥ 0 µg/m³
     - O3: ≥ 0 µg/m³
     - SO2: ≥ 0 µg/m³
   - **Astronomical**:
     - Sunrise/Sunset: Valid time (HH:MM)
     - Moonrise/Moonset: Valid time (HH:MM) or NULL if no rise/set
     - Moon phase: 0-1
     - Day length: 0-24 hours
4. **No duplicates**: Each (city, timestamp) pair should be unique
5. **Astronomical consistency**: Sunrise/sunset times should be same for all hours in a day
    "Bengaluru": (12.9716, 77.5946)
}

data = []

for city, (lat, lon) in cities.items():
    # Get historical data
    # (Implementation in scripts/download_imd_data.py)
    pass
```

## Data Quality Checks

Before using your data, ensure:

1. **No missing values**: Fill gaps with interpolation
2. **Consistent timestamps**: Hourly intervals
3. **Valid ranges**:
   - Temperature: -10°C to 50°C (India)
   - Humidity: 0-100%
   - Wind speed: ≥ 0 km/h
   - Rainfall: ≥ 0 mm
   - Pressure: 900-1100 hPa
   - Cloud cover: 0-100%
4. **No duplicates**: Each (city, timestamp) pair should be unique

## Recommended Approach

**For Quick Start** (Testing):
1. Use sample_data.csv provided
2. Or download 1 month of data from OpenWeatherMap free tier
3. Train on small dataset to verify pipeline

**For Production** (Best Results):
1. **Option A**: Purchase historical data from Visual Crossing or World Weather Online
   - Cost: ~$50-100 for 2 years of data
   - Fastest and most reliable

2. **Option B**: Collect via free APIs over time
   - Set up automated script to collect hourly
   - Run for 6+ months to build dataset
   - Free but time-consuming

3. **Option C**: Request from IMD
   - Official government data
   - Most accurate for India
   - May require paperwork/payment

## Next Steps

1. Choose your data source
2. Collect/purchase data
3. Format according to template (see `weather_time_series_template.csv`)
4. Validate using `scripts/validate_data.py`
5. Start training!

## Need Help?

If you encounter issues:
- Check the validation script output
- Ensure all columns match the template exactly
- Verify timestamp format: `YYYY-MM-DD HH:MM`
- Make sure there are no gaps in hourly data
