# Data Collection Guide for Mausam-Vaani

## Overview
This guide explains how to collect and prepare weather data for training the TFT model.

## Required Data Format

The model requires **hourly weather data** in CSV format with the following columns:

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

## Minimum Data Requirements

- **Duration**: At least **6 months** of continuous data (ideally 1-2 years)
- **Frequency**: **Hourly** measurements (one row per hour)
- **Cities**: At least **5-10 major Indian cities** for better generalization
- **Total Records**: Minimum ~43,800 rows (6 months × 730 hours/month × 10 cities)

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

### 2. OpenWeatherMap
**Website**: https://openweathermap.org/

**Access**:
- Free tier: 1,000 calls/day
- Paid plans for historical data
- API documentation: https://openweathermap.org/api

**How to use**:
```python
import requests

API_KEY = "your_api_key"
city = "Delhi"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"

response = requests.get(url)
data = response.json()
```

### 3. Visual Crossing Weather
**Website**: https://www.visualcrossing.com/

**Access**:
- Free tier: 1,000 records/day
- Historical weather data API
- Good for Indian cities

**How to use**:
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
2. Filter by "Ministry of Earth Sciences"
3. Download available datasets

### 5. World Weather Online (Premium)
**Website**: https://www.worldweatheronline.com/

**Features**:
- Historical weather API
- Very comprehensive data
- Paid service but reasonable rates

## Data Collection Script Example

We've provided scripts to help you collect data:

### Using OpenWeatherMap (see `scripts/download_imd_data.py`)

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "your_key"
cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
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
