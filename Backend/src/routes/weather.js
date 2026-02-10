const express = require('express');
const { body, validationResult } = require('express-validator');
const WeatherReport = require('../models/WeatherReport');
const User = require('../models/User');
const { protect, optionalAuth } = require('../middleware/auth');

const router = express.Router();

// ===========================================
// @route   GET /api/weather/current
// @desc    Get current weather + advice (no auth needed)
// @access  Public
// ===========================================
router.get('/current', optionalAuth, async (req, res) => {
  try {
    const { lat, lon, city } = req.query;

    // Get user persona for personalized advice
    const persona = req.user?.persona || 'general';
    const language = req.user?.language || 'en';

    // Mock weather data - in production, call actual weather API
    const weather = generateMockWeather(lat, lon, city);
    
    // Generate advice based on weather and persona
    const advice = generateAdvice(weather, persona, language);
    
    // Get nearby community reports
    const reports = await WeatherReport.find({
      'location.coordinates.latitude': { $gte: parseFloat(lat) - 0.1, $lte: parseFloat(lat) + 0.1 },
      'location.coordinates.longitude': { $gte: parseFloat(lon) - 0.1, $lte: parseFloat(lon) + 0.1 },
      createdAt: { $gte: new Date(Date.now() - 3 * 60 * 60 * 1000) }, // Last 3 hours
    }).limit(5).sort({ createdAt: -1 });

    res.json({
      success: true,
      data: {
        weather,
        advice,
        communityReports: reports,
        personalized: !!req.user,
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to fetch weather',
      error: error.message,
    });
  }
});

// ===========================================
// @route   GET /api/weather/hourly
// @desc    Get hourly forecast
// @access  Public
// ===========================================
router.get('/hourly', async (req, res) => {
  try {
    const { lat, lon } = req.query;

    // Mock hourly data
    const hourly = [];
    const now = new Date();
    
    for (let i = 0; i < 24; i++) {
      const hour = new Date(now.getTime() + i * 60 * 60 * 1000);
      hourly.push({
        time: hour.toISOString(),
        hour: hour.getHours(),
        temp: Math.round(25 + Math.random() * 10),
        condition: ['sunny', 'cloudy', 'partly_cloudy', 'rain'][Math.floor(Math.random() * 4)],
        rainChance: Math.round(Math.random() * 100),
        humidity: Math.round(50 + Math.random() * 40),
      });
    }

    res.json({
      success: true,
      data: hourly,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to fetch hourly forecast',
    });
  }
});

// ===========================================
// @route   GET /api/weather/risks
// @desc    Get risk assessment
// @access  Public
// ===========================================
router.get('/risks', optionalAuth, async (req, res) => {
  try {
    const { lat, lon } = req.query;
    const persona = req.user?.persona || 'general';

    // Mock risk data
    const risks = {
      rain: {
        level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
        probability: Math.round(Math.random() * 100),
        message: 'Rain expected after 3 PM',
      },
      heat: {
        level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
        index: Math.round(30 + Math.random() * 15),
        message: 'Stay hydrated',
      },
      aqi: {
        level: ['good', 'moderate', 'poor', 'unhealthy'][Math.floor(Math.random() * 4)],
        value: Math.round(50 + Math.random() * 200),
        message: 'Air quality is moderate',
      },
      wind: {
        level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
        speed: Math.round(5 + Math.random() * 30),
        message: 'Moderate winds expected',
      },
    };

    res.json({
      success: true,
      data: risks,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to fetch risks',
    });
  }
});

// ===========================================
// @route   POST /api/weather/report
// @desc    Submit community weather report
// @access  Private
// ===========================================
router.post('/report', protect, [
  body('reportType').isIn(['rain', 'no_rain', 'flood', 'heatwave', 'storm', 'fog', 'clear', 'other']),
  body('latitude').isFloat(),
  body('longitude').isFloat(),
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }

    const { reportType, intensity, description, latitude, longitude, city, state } = req.body;

    const report = await WeatherReport.create({
      user: req.user._id,
      location: {
        city: city || 'Unknown',
        state: state || '',
        coordinates: {
          latitude,
          longitude,
        },
      },
      reportType,
      intensity: intensity || 'moderate',
      description,
    });

    // Add points to user
    await User.findByIdAndUpdate(req.user._id, {
      $inc: { points: 5, weatherReports: 1 },
    });

    res.status(201).json({
      success: true,
      message: 'Report submitted! +5 points',
      data: report,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to submit report',
      error: error.message,
    });
  }
});

// ===========================================
// @route   GET /api/weather/plan-day
// @desc    Get AI day planner advice
// @access  Public
// ===========================================
router.get('/plan-day', optionalAuth, async (req, res) => {
  try {
    const { lat, lon, activity, time } = req.query;
    const persona = req.user?.persona || 'general';

    // Mock day plan
    const plan = {
      bestTimes: ['9:00 AM - 12:00 PM', '5:00 PM - 7:00 PM'],
      avoidTimes: ['2:00 PM - 4:00 PM'],
      recommendation: 'Morning is best for outdoor activities. Avoid afternoon due to heat.',
      warnings: [],
      tips: [
        'Carry water bottle',
        'Wear light clothes',
        'Check weather at 2 PM for updates',
      ],
    };

    // Add rain warning if applicable
    if (Math.random() > 0.5) {
      plan.warnings.push('Rain expected after 3 PM. Carry umbrella.');
    }

    res.json({
      success: true,
      data: plan,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to generate plan',
    });
  }
});

// ===========================================
// Helper Functions
// ===========================================

function generateMockWeather(lat, lon, city) {
  const conditions = ['sunny', 'cloudy', 'partly_cloudy', 'rain', 'thunderstorm'];
  const condition = conditions[Math.floor(Math.random() * 3)]; // Bias towards good weather
  
  return {
    location: {
      city: city || 'Unknown',
      lat: parseFloat(lat) || 23.2599,
      lon: parseFloat(lon) || 77.4126,
    },
    current: {
      temp: Math.round(25 + Math.random() * 12),
      feelsLike: Math.round(26 + Math.random() * 12),
      humidity: Math.round(50 + Math.random() * 40),
      windSpeed: Math.round(5 + Math.random() * 20),
      condition,
      icon: getWeatherIcon(condition),
      aqi: Math.round(50 + Math.random() * 150),
    },
    today: {
      high: Math.round(32 + Math.random() * 8),
      low: Math.round(20 + Math.random() * 5),
      rainChance: Math.round(Math.random() * 60),
      sunrise: '6:15 AM',
      sunset: '6:45 PM',
    },
    updatedAt: new Date().toISOString(),
  };
}

function getWeatherIcon(condition) {
  const icons = {
    sunny: '☀️',
    cloudy: '☁️',
    partly_cloudy: '⛅',
    rain: '🌧️',
    thunderstorm: '⛈️',
    fog: '🌫️',
    clear: '🌙',
  };
  return icons[condition] || '☀️';
}

function generateAdvice(weather, persona, language) {
  const temp = weather.current.temp;
  const condition = weather.current.condition;
  const rainChance = weather.today.rainChance;
  const aqi = weather.current.aqi;

  let mainAdvice = '';
  let safetyLevel = 'safe';
  const tips = [];

  // Main advice based on condition
  if (condition === 'rain' || condition === 'thunderstorm') {
    mainAdvice = 'Rain expected. Carry umbrella and avoid travel if possible.';
    safetyLevel = 'caution';
    tips.push('Avoid waterlogging areas');
  } else if (rainChance > 50) {
    mainAdvice = `Rain likely after afternoon. ${rainChance}% chance. Plan accordingly.`;
    safetyLevel = 'caution';
    tips.push('Keep umbrella handy');
  } else if (temp > 38) {
    mainAdvice = 'Extreme heat today. Avoid outdoor activities 12-4 PM.';
    safetyLevel = 'warning';
    tips.push('Stay hydrated', 'Wear light clothes');
  } else if (aqi > 150) {
    mainAdvice = 'Poor air quality. Limit outdoor exposure.';
    safetyLevel = 'warning';
    tips.push('Wear mask outdoors', 'Keep windows closed');
  } else {
    mainAdvice = 'Good day for outdoor activities. No major risks.';
    safetyLevel = 'safe';
    tips.push('Enjoy your day!');
  }

  // Persona-specific tips
  if (persona === 'driver') {
    if (condition === 'rain') tips.push('Drive slowly, roads may be slippery');
    if (temp > 35) tips.push('Check tire pressure in heat');
  } else if (persona === 'farmer') {
    if (rainChance > 30) tips.push('Consider delaying irrigation');
    if (condition === 'sunny') tips.push('Good day for harvesting');
  } else if (persona === 'student') {
    if (condition === 'rain') tips.push('Leave early for school');
  }

  return {
    main: mainAdvice,
    safetyLevel,
    tips,
    summary: safetyLevel === 'safe' ? '✅ Safe day' : safetyLevel === 'caution' ? '⚠️ Some risks' : '🚨 Be careful',
  };
}

module.exports = router;
