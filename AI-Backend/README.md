# Mausam-Vaani AI Backend

Complete AI-powered hyperlocal weather prediction system with **Temporal Fusion Transformer (TFT)** deep learning model and **Gemini LLM** integration for personalized insights.

## 🌟 Features

- **Hyperlocal Weather Prediction**: TFT model predicts weather up to 168 hours (7 days) ahead at street/village level
- **Multi-Variable Forecasting**: Predicts temperature, humidity, wind speed, rainfall, pressure, and cloud cover
- **Personalized Insights**: Gemini AI generates profession-specific advice for farmers, commuters, construction workers, etc.
- **REST API**: Flask-based API for easy integration with frontend
- **Comprehensive Training Pipeline**: Data preprocessing, training, evaluation, and model serving
- **Scalable Architecture**: Supports batch predictions and real-time inference

## 📁 Project Structure

```
AI-Backend/
├── models/                    # Deep Learning Models
│   ├── tft_model.py          # Temporal Fusion Transformer architecture
│   ├── data_preprocessing.py # Data pipeline & feature engineering
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation metrics & visualization
│   └── model_serving.py      # Production inference
│
├── api/                       # Flask REST API
│   ├── app.py                # Main Flask application
│   ├── routes.py             # API endpoints
│   ├── gemini_integration.py # Gemini AI integration
│   └── utils.py              # Validation utilities
│
├── config/                    # Configuration Files
│   ├── model_config.yaml     # Model hyperparameters
│   └── .env.example          # Environment variables template
│
├── data/                      # Data Directory
│   ├── sample_data.csv       # Sample weather data
│   ├── weather_time_series_template.csv  # Data format template
│   └── data_collection_guide.md         # Data collection guide
│
├── scripts/                   # Utility Scripts
│   └── validate_data.py      # Data validation tool
│
├── checkpoints/              # Model Checkpoints (created during training)
├── logs/                     # TensorBoard Logs (created during training)
├── results/                  # Evaluation Results (created during evaluation)
│
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to AI-Backend directory
cd "c:\personal dg\github_repo\Mausam-Vaani-\AI-Backend"

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
copy config\.env.example .env

# Edit .env file and add:
# - GEMINI_API_KEY (get from https://aistudio.google.com/app/apikey)
# - MODEL_PATH (default: checkpoints/best_model.pth)
# - Other configurations as needed
```

### 3. Prepare Data

**Option A**: Use sample data (for testing):
```bash
# Sample data is already provided in data/sample_data.csv
# Note: This is insufficient for training, only for testing API
```

**Option B**: Collect real data:
```bash
# Follow the guide in data/data_collection_guide.md
# Place your CSV file as: data/weather_time_series.csv

# Validate your data
python scripts/validate_data.py --input data/weather_time_series.csv
```

### 4. Train Model

```bash
# Train with default configuration
python models/train.py --config config/model_config.yaml

# Or specify number of epochs
python models/train.py --config config/model_config.yaml --epochs 50

# Resume from checkpoint
python models/train.py --resume checkpoints/latest_checkpoint.pth
```

**Training on Google Colab** (if no GPU locally):
1. Upload all files to Google Drive
2. Open Colab notebook (create one)
3. Mount Drive and install requirements
4. Run training script

### 5. Run Flask API

```bash
# Make sure .env is configured
# Make sure model is trained (checkpoints/best_model.pth exists)

# Start API server
python api/app.py

# Server will start on http://localhost:5000
```

### 6. Test API

**Check health**:
```bash
curl http://localhost:5000/health
```

**Predict weather** (PowerShell):
```powershell
$headers = @{"Content-Type"="application/json"}
$body = @{
    historical_data = @{
        timestamp = @("2024-11-23 00:00", "2024-11-23 01:00", ...)  # 168 hours
        temperature = @(25.0, 24.8, ...)
        humidity = @(65, 66, ...)
        wind_speed = @(5.2, 5.0, ...)
        rainfall = @(0.0, 0.0, ...)
        pressure = @(1010, 1009, ...)
        cloud_cover = @(20, 25, ...)
        latitude = @(28.6139, 28.6139, ...)
        longitude = @(77.2090, 77.2090, ...)
    }
    forecast_steps = 24
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:5000/api/predict-weather" -Method POST -Headers $headers -Body $body
```

**Get personalized insight**:
```powershell
$body = @{
    latitude = 28.6139
    longitude = 77.2090
    city = "Delhi"
    user_profession = "Farmer"
    user_context = @{
        crop = "Rice"
    }
    historical_data = @{ ... }  # Same as above
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:5000/api/get-insight" -Method POST -Headers $headers -Body $body
```

## 📊 Model Details

### Temporal Fusion Transformer (TFT)

**Architecture**:
- **Input**: 168 hours (1 week) of historical weather data
- **Output**: 24-168 hours future weather predictions
- **Features**: 15 input features (weather params + time encodings)
- **Targets**: 6 output variables (temperature, humidity, wind speed, rainfall, pressure, cloud cover)

**Components**:
1. **Variable Selection Network**: Learns feature importance
2. **GRN (Gated Residual Network)**: Feature processing with gating
3. **LSTM Encoder-Decoder**: Temporal sequence modeling
4. **Multi-Head Attention**: Captures long-range dependencies
5. **Output Layer**: Multi-step forecasting

**Hyperparameters** (default):
- Hidden dimension: 128
- Number of attention heads: 4
- LSTM layers: 2
- Dropout: 0.1
- Learning rate: 0.001

## 🗂️ Data Requirements

### Format
CSV file with hourly weather data containing:

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| timestamp | datetime | - | Date and time (YYYY-MM-DD HH:MM) |
| city | string | - | City name |
| latitude | float | degrees | Latitude coordinate |
| longitude | float | degrees | Longitude coordinate |
| temperature | float | °C | Air temperature |
| humidity | float | % | Relative humidity (0-100) |
| wind_speed | float | km/h | Wind speed |
| rainfall | float | mm | Rainfall amount |
| pressure | float | hPa | Atmospheric pressure |
| cloud_cover | float | % | Cloud coverage (0-100) |

### Minimum Requirements
- **Duration**: 6 months (ideally 1-2 years)
- **Frequency**: Hourly measurements
- **Cities**: 5-10 major Indian cities
- **Total Records**: ~43,800+ rows

### Data Sources
1. **IMD (India Meteorological Department)**: https://mausam.imd.gov.in/
2. **OpenWeatherMap**: https://openweathermap.org/
3. **Visual Crossing**: https://www.visualcrossing.com/
4. **Government Open Data**: https://data.gov.in/

See `data/data_collection_guide.md` for detailed instructions.

## 🔌 API Endpoints

### POST `/api/predict-weather`
Get weather predictions for next N hours.

**Request**:
```json
{
  "historical_data": {
    "timestamp": ["2024-11-23 00:00", ...],
    "temperature": [25.0, ...],
    "humidity": [65, ...],
    ...
  },
  "forecast_steps": 24
}
```

**Response**:
```json
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
```

### POST `/api/get-insight`
Get personalized weather insights with Gemini AI.

**Request**:
```json
{
  "latitude": 28.6139,
  "longitude": 77.2090,
  "city": "Delhi",
  "user_profession": "Farmer",
  "user_context": {
    "crop": "Rice"
  },
  "historical_data": { ... },
  "forecast_steps": 24
}
```

**Response**:
```json
{
  "success": true,
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "city": "Delhi"
  },
  "weather": {
    "current": {
      "temperature": 28,
      "humidity": 85,
      ...
    },
    "condition": "Heavy Rain"
  },
  "personalized_insight": "🌧️ Due to heavy rainfall, avoid pesticide spraying today. You can start preparing your field for rice sowing."
}
```

### GET `/health`
Health check endpoint.

## 🎯 Next Steps for You

### Immediate Actions

1. **Get Gemini API Key**:
   - Visit https://aistudio.google.com/app/apikey
   - Create/sign in with Google account
   - Generate API key
   - Add to `.env` file

2. **Obtain Training Data**:
   - Review `data/data_collection_guide.md`
   - Choose a data source (Visual Crossing recommended for quick start)
   - Collect at least 6 months of data for 5-10 Indian cities
   - Place in `data/weather_time_series.csv`

3. **Validate Data**:
   ```bash
   python scripts/validate_data.py --input data/weather_time_series.csv
   ```

4. **Train Model**:
   ```bash
   python models/train.py --config config/model_config.yaml --epochs 50
   ```
   - Watch TensorBoard for training progress:
     ```bash
     tensorboard --logdir logs
     ```

5. **Test API**:
   - Start Flask server
   - Test with sample requests
   - Verify predictions are reasonable

6. **Integrate with Frontend**:
   - Update your React frontend to call API endpoints
   - Handle weather prediction and insight responses
   - Display data in UI

### Optional Enhancements

- **Add more cities**: Collect data for more locations
- **Extend forecast horizon**: Train for longer predictions (up to 168 hours)
- **Fine-tune model**: Experiment with hyperparameters in `config/model_config.yaml`
- **Add more professions**: Extend Gemini prompts in `api/gemini_integration.py`
- **Deploy to cloud**: Deploy Flask API to Railway, Render, or AWS

## 🐛 Troubleshooting

### Model Training Issues

**Out of Memory**:
- Reduce `batch_size` in `config/model_config.yaml`
- Reduce `hidden_dim` (e.g., 64 instead of 128)
- Use Google Colab with GPU

**Poor Performance**:
- Check data quality with validation script
- Train for more epochs
- Ensure sufficient training data (6+ months)
- Increase model capacity (larger `hidden_dim`)

### API Issues

**Model not found**:
- Ensure model is trained: `checkpoints/best_model.pth` exists
- Check `MODEL_PATH` in `.env`

**Gemini API error**:
- Verify `GEMINI_API_KEY` is correct in `.env`
- Check API quota/limits
- Fallback insights will be used if Gemini fails

**CORS errors**:
- Check frontend URL is in CORS origins in `api/app.py`
- Verify frontend is making requests to correct endpoint

## 📝 License

MIT License - feel free to use for your project!

## 👨‍💻 Development Notes

**Created**: November 2024  
**Model**: Temporal Fusion Transformer (TFT)  
**Framework**: PyTorch + Flask  
**LLM**: Google Gemini 1.5 Flash  

---

**Need Help?** Check the documentation files:
- `data/data_collection_guide.md` - Data collection instructions
- `config/model_config.yaml` - Model configuration
- `api/routes.py` - API endpoint details
