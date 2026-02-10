const express = require('express');
const router = express.Router();

/**
 * AI Routes for Mausam Vaani
 * Two endpoints as per product architecture:
 * 1. Quick Insight - Fast, lightweight dashboard advice
 * 2. Planner - Comprehensive scenario-based predictions
 */

// ===========================================
// QUICK INSIGHT ENDPOINT
// ===========================================

/**
 * POST /api/ai/quick-insight
 * Fast micro-advice for dashboard (<200ms target)
 * 
 * Request body:
 * {
 *   weatherData: { current, hourly, daily },
 *   persona: 'driver' | 'farmer' | 'worker' | etc.,
 *   location: { lat, lon, city },
 *   weatherRisks: ['heat', 'rain', etc.],
 *   timestamp: ISO string
 * }
 */
router.post('/quick-insight', async (req, res) => {
  try {
    const { weatherData, persona = 'general', location, weatherRisks = [] } = req.body;
    
    if (!weatherData) {
      return res.status(400).json({
        success: false,
        message: 'Weather data is required'
      });
    }

    const insight = generateQuickInsight(weatherData, persona, weatherRisks);
    
    res.json({
      success: true,
      ...insight
    });

  } catch (error) {
    console.error('Quick insight error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate insight'
    });
  }
});

// ===========================================
// PLANNER ENDPOINT
// ===========================================

/**
 * POST /api/ai/planner
 * Comprehensive scenario-based predictions
 * 
 * Request body:
 * {
 *   persona,
 *   location,
 *   weatherData,
 *   activity: 'travel' | 'farming' | 'outdoor' | etc.,
 *   date: 'YYYY-MM-DD',
 *   timeRange: { start: 'HH:MM', end: 'HH:MM' },
 *   risks: ['avoid_rain', 'avoid_heat', etc.],
 *   duration: 1-10 hours,
 *   notes: string
 * }
 */
router.post('/planner', async (req, res) => {
  try {
    const { 
      persona = 'general',
      location,
      weatherData,
      activity,
      date,
      timeRange,
      risks = [],
      duration,
      notes
    } = req.body;

    if (!activity || !weatherData) {
      return res.status(400).json({
        success: false,
        message: 'Activity and weather data are required'
      });
    }

    const plan = generateSmartPlan({
      persona,
      location,
      weatherData,
      activity,
      date,
      timeRange,
      risks,
      duration,
      notes
    });

    res.json({
      success: true,
      ...plan
    });

  } catch (error) {
    console.error('Planner error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate plan'
    });
  }
});

// ===========================================
// AI LOGIC FUNCTIONS
// ===========================================

/**
 * Generate quick insight for dashboard
 */
function generateQuickInsight(weather, persona, weatherRisks = []) {
  const current = weather?.current || {};
  const temp = current.temp || 25;
  const humidity = current.humidity || 50;
  const condition = (current.condition || '').toLowerCase();
  const wind = current.wind || 0;
  
  let title = 'Today\'s Tip';
  let message = '';
  const tips = [];

  // Temperature-based advice
  if (temp >= 35) {
    message = 'Extreme heat! Stay hydrated and avoid outdoor activities 12-4pm';
    title = 'Heat Alert';
    tips.push('🔥 ' + message);
  } else if (temp >= 30) {
    tips.push('☀️ High temperature. Carry water and wear light clothes');
  } else if (temp <= 15) {
    tips.push('🧥 Cold weather. Layer up before heading out');
  }

  // Rain advice
  if (condition.includes('rain') || condition.includes('drizzle')) {
    message = message || 'Carry umbrella. Roads may be slippery';
    title = condition.includes('heavy') ? 'Heavy Rain' : 'Rain Expected';
    tips.push('🌧️ ' + message);
  }

  // Humidity advice
  if (humidity >= 80) {
    tips.push('💧 High humidity. Take breaks if working outdoors');
  }

  // Wind advice
  if (wind >= 20) {
    tips.push('💨 Strong winds expected. Secure loose items');
    if (!message) {
      message = 'Strong winds expected. Secure loose items';
      title = 'Wind Advisory';
    }
  }

  // Fog advice
  if (condition.includes('fog') || condition.includes('mist')) {
    message = 'Low visibility. Drive carefully';
    title = 'Fog Alert';
    tips.push('🌫️ ' + message);
  }

  // Persona-specific advice
  if (persona === 'driver' || persona === 'delivery') {
    if (condition.includes('rain')) {
      tips.push('🚗 Increase following distance on wet roads');
    }
    if (condition.includes('fog')) {
      tips.push('🚗 Use fog lights and reduce speed');
    }
  } else if (persona === 'farmer') {
    if (condition.includes('rain')) {
      tips.push('🌾 Good day for field irrigation. Postpone fertilizer application');
    }
    if (temp >= 32) {
      tips.push('🌾 Irrigate crops in early morning or evening');
    }
  } else if (persona === 'worker') {
    if (temp >= 32) {
      tips.push('👷 Take frequent breaks in shade. Hydrate every 30 mins');
    }
  }

  // Best time suggestion
  const hour = new Date().getHours();
  if (temp >= 30 && hour < 10) {
    tips.push('⏰ Best outdoor window: Now until 10 AM');
  } else if (temp >= 30) {
    tips.push('⏰ Better conditions expected after 5 PM');
  }

  // Set default message if none set
  if (!message) {
    message = tips[0]?.replace(/^[^\s]+\s/, '') || 'Good weather conditions for your activities today.';
  }

  return {
    title,
    message,
    tips,
    persona,
    generatedAt: new Date().toISOString()
  };
}

/**
 * Generate comprehensive smart plan
 */
function generateSmartPlan(data) {
  const { weatherData, activity, date, timeRange, risks = [], persona, duration, notes } = data;
  const current = weatherData?.current || {};
  const temp = current.temp || 25;
  const condition = (current.condition || '').toLowerCase();
  const wind = current.wind || 0;
  const humidity = current.humidity || 50;

  const tips = [];
  let riskLevel = 'Low';
  let recommendation = 'Good conditions for your activity';
  let bestTime = timeRange?.start || '09:00';
  let avoidTime = null;

  // Activity-specific analysis
  switch (activity) {
    case 'travel':
    case 'commute':
      if (condition.includes('rain')) {
        tips.push('Allow extra travel time due to wet roads');
        tips.push('Check traffic updates before departure');
        riskLevel = 'Medium';
        recommendation = 'Travel possible but allow extra time for wet conditions';
      }
      if (condition.includes('fog')) {
        riskLevel = 'High';
        recommendation = 'Delay travel if possible. Visibility is low';
        tips.push('Use fog lights and maintain safe distance');
      }
      if (temp >= 35) {
        tips.push('Ensure AC is working. Carry water');
        recommendation = 'Avoid peak afternoon hours (12-4 PM)';
        avoidTime = '12:00 - 16:00';
        bestTime = '06:00 - 10:00';
      }
      break;

    case 'outdoor':
    case 'exercise':
      if (temp >= 32) {
        recommendation = 'Schedule for early morning (6-9 AM) or evening (5-7 PM)';
        bestTime = '06:00 - 09:00';
        avoidTime = '11:00 - 16:00';
        tips.push('Carry water and electrolytes');
        tips.push('Wear light, breathable clothing');
        riskLevel = 'Medium';
      }
      if (condition.includes('rain')) {
        tips.push('Consider indoor alternatives');
        riskLevel = 'High';
        recommendation = 'Not recommended for outdoor exercise today';
      }
      if (humidity >= 80) {
        tips.push('High humidity - take frequent breaks');
      }
      break;

    case 'farming':
      if (condition.includes('rain')) {
        recommendation = 'Good day for indoor farm work. Avoid field operations';
        tips.push('Postpone pesticide and fertilizer application');
        tips.push('Check drainage systems');
        riskLevel = 'Medium';
      }
      if (temp >= 35) {
        recommendation = 'Work in early morning (5-10 AM) or late evening';
        bestTime = '05:00 - 10:00';
        avoidTime = '11:00 - 17:00';
        tips.push('Hydrate workers frequently');
        tips.push('Provide shade breaks');
        riskLevel = 'Medium';
      }
      break;

    case 'event':
      if (condition.includes('rain')) {
        riskLevel = 'High';
        recommendation = 'Have backup indoor venue ready';
        tips.push('Arrange canopy/tent coverage');
        tips.push('Prepare rain contingency plan');
      }
      if (temp >= 35) {
        tips.push('Arrange shade and cooling stations');
        tips.push('Provide water stations for guests');
        riskLevel = riskLevel === 'High' ? 'High' : 'Medium';
      }
      if (wind >= 25) {
        tips.push('Secure all decorations and banners');
        riskLevel = 'Medium';
      }
      break;

    case 'delivery':
      if (condition.includes('rain')) {
        tips.push('Protect packages from water damage');
        tips.push('Allow extra delivery time');
        riskLevel = 'Medium';
        recommendation = 'Deliveries possible but plan for delays';
      }
      if (temp >= 35) {
        tips.push('Prioritize AC in vehicle during peak hours');
        tips.push('Take breaks in shade');
        avoidTime = '13:00 - 16:00';
      }
      break;

    default:
      if (temp >= 32) {
        tips.push('Stay hydrated and avoid peak sun hours');
        avoidTime = '12:00 - 16:00';
      }
      if (condition.includes('rain')) {
        tips.push('Carry rain protection');
      }
  }

  // Risk-based adjustments
  if (risks.includes('avoid_rain') && condition.includes('rain')) {
    riskLevel = 'High';
    recommendation = 'High rain probability. Consider postponing outdoor activities';
  }
  if (risks.includes('avoid_heat') && temp >= 32) {
    riskLevel = riskLevel === 'High' ? 'High' : 'Medium';
    recommendation = 'High temperature expected. Best hours: before 10 AM or after 5 PM';
    bestTime = '06:00 - 10:00';
    avoidTime = '12:00 - 17:00';
  }
  if (risks.includes('avoid_wind') && wind >= 20) {
    tips.push('Strong winds - take precautions');
    riskLevel = riskLevel === 'High' ? 'High' : 'Medium';
  }
  if (risks.includes('avoid_fog') && condition.includes('fog')) {
    riskLevel = 'High';
    recommendation = 'Low visibility conditions. Delay outdoor activities if possible';
  }

  // Notes-based adjustments
  if (notes) {
    const notesLower = notes.toLowerCase();
    if (notesLower.includes('elderly') || notesLower.includes('senior')) {
      tips.push('Extra care needed for elderly - avoid extreme conditions');
    }
    if (notesLower.includes('kids') || notesLower.includes('children')) {
      tips.push('Keep children hydrated and protected from sun');
    }
    if (notesLower.includes('equipment')) {
      tips.push('Protect equipment from weather exposure');
    }
  }

  // Default tips if none generated
  if (tips.length === 0) {
    tips.push('Weather conditions are favorable');
    tips.push('Stay aware of any changes in forecast');
  }

  return {
    recommendation,
    bestTime,
    avoidTime,
    riskLevel,
    tips,
    activity,
    date,
    generatedAt: new Date().toISOString()
  };
}

module.exports = router;
