import api from "./api";

// AI Backend URL (FastAPI)
const AI_BACKEND_URL = import.meta.env.VITE_AI_BACKEND_URL || "http://localhost:8000";

/**
 * AI Insight service — provides quick insights and comprehensive planning.
 * Two types:
 * 1. Quick Insight (Dashboard) - fast, lightweight advice
 * 2. Smart Planner - comprehensive scenario-based predictions
 * 
 * Calls FastAPI AI-Backend for Gemini-powered insights
 */
export const insightService = {
  /**
   * Quick insight for dashboard - lightweight, fast
   * Returns micro-advice like "Carry umbrella", "Avoid 2-4pm heat"
   * @param {Object} data - Object containing weatherData, persona, location, weatherRisks
   */
  async getQuickInsight(data) {
    const { weatherData, persona = 'general', location = null, weatherRisks = [] } = data || {};
    
    try {
      // Call FastAPI AI-Backend
      const response = await fetch(`${AI_BACKEND_URL}/quick-insight`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          weatherData,
          persona,
          location,
          weatherRisks,
          timestamp: new Date().toISOString()
        })
      });
      
      if (!response.ok) {
        throw new Error(`AI Backend error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.warn('AI Backend quick-insight failed:', error.message);
      // Fallback to local smart advice if API fails
      return this.generateLocalInsight(weatherData, persona, weatherRisks);
    }
  },

  /**
   * Comprehensive AI planner - scenario-based predictions
   * Takes activity, date, time range, risks, and returns smart recommendations
   * Calls FastAPI AI-Backend with Gemini integration
   */
  async getSmartPlan(plannerData) {
    try {
      // Call FastAPI AI-Backend
      const response = await fetch(`${AI_BACKEND_URL}/planner`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(plannerData)
      });
      
      if (!response.ok) {
        throw new Error(`AI Backend error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.warn('AI Backend planner failed:', error.message);
      // Fallback to rule-based advice
      return this.generateLocalPlan(plannerData);
    }
  },

  /**
   * Legacy method for backward compatibility
   */
  async getInsight(weatherData, persona = {}) {
    return this.getQuickInsight(weatherData, persona?.persona || 'general');
  },

  /**
   * Legacy planner advice method
   */
  async getPlannerAdvice(weatherData, date, activity) {
    return this.getSmartPlan({
      weatherData,
      date,
      activity,
      timeRange: { start: '09:00', end: '18:00' }
    });
  },

  /**
   * Generate local insight when API is unavailable
   * Rule-based smart advice based on weather conditions
   */
  generateLocalInsight(weather, persona, weatherRisks = []) {
    const insights = [];
    const current = weather?.current || {};
    const temp = current.temp || 25;
    const humidity = current.humidity || 50;
    const condition = (current.condition || '').toLowerCase();
    const wind = current.wind || 0;
    
    let title = 'Smart Advice';
    let primaryMessage = '';

    // Temperature-based advice
    if (temp >= 35) {
      primaryMessage = "Extreme heat! Stay hydrated and avoid outdoor activities 12-4pm";
      title = "Heat Alert";
      insights.push("🔥 " + primaryMessage);
    } else if (temp >= 30) {
      insights.push("☀️ High temperature. Carry water and wear light clothes");
    } else if (temp <= 15) {
      insights.push("🧥 Cold weather. Layer up before heading out");
    }

    // Rain advice
    if (condition.includes('rain') || condition.includes('drizzle')) {
      primaryMessage = primaryMessage || "Carry umbrella. Roads may be slippery";
      title = condition.includes('heavy') ? "Heavy Rain" : "Rain Expected";
      insights.push("🌧️ " + primaryMessage);
    }

    // Humidity advice
    if (humidity >= 80) {
      insights.push("💧 High humidity. Take breaks if working outdoors");
    }

    // Wind advice
    if (wind >= 20) {
      insights.push("💨 Strong winds expected. Secure loose items");
      if (!primaryMessage) {
        primaryMessage = "Strong winds expected. Secure loose items";
        title = "Wind Advisory";
      }
    }

    // Fog advice
    if (condition.includes('fog') || condition.includes('mist')) {
      primaryMessage = "Low visibility. Drive carefully";
      title = "Fog Alert";
      insights.push("🌫️ " + primaryMessage);
    }

    // Persona-specific advice
    if (persona === 'driver' || persona === 'delivery') {
      if (condition.includes('rain')) {
        insights.push("🚗 Increase following distance on wet roads");
      }
    } else if (persona === 'farmer') {
      if (condition.includes('rain')) {
        insights.push("🌾 Good day for field irrigation. Postpone fertilizer application");
      }
    } else if (persona === 'worker') {
      if (temp >= 32) {
        insights.push("👷 Take frequent breaks in shade. Hydrate every 30 mins");
      }
    }

    // Best time suggestion
    const hour = new Date().getHours();
    if (temp >= 30 && hour < 10) {
      insights.push("⏰ Best outdoor window: Now until 10 AM");
    } else if (temp >= 30) {
      insights.push("⏰ Better conditions expected after 5 PM");
    }

    // Set default message if none set
    if (!primaryMessage) {
      primaryMessage = insights[0]?.replace(/^[^\s]+\s/, '') || "Good weather conditions for your activities today.";
      title = 'Today\'s Tip';
    }

    return {
      success: true,
      title: title,
      message: primaryMessage,
      tips: insights,
      source: 'local'
    };
  },

  /**
   * Generate local plan when API is unavailable
   */
  generateLocalPlan(plannerData) {
    const { weatherData, activity, date, timeRange, risks = [] } = plannerData;
    const current = weatherData?.current || {};
    const temp = current.temp || 25;
    const condition = (current.condition || '').toLowerCase();
    const hourly = weatherData?.hourly || [];

    const tips = [];
    let riskLevel = 'Low';
    let recommendation = 'Good conditions for your activity';
    let bestTime = timeRange?.start || '09:00';
    let avoidTime = null;

    // Activity-specific analysis
    if (activity === 'travel' || activity === 'commute') {
      if (condition.includes('rain')) {
        tips.push("Allow extra travel time due to wet roads");
        tips.push("Check traffic updates before departure");
        riskLevel = 'Medium';
      }
      if (temp >= 35) {
        tips.push("Ensure AC is working. Carry water");
        recommendation = "Avoid peak afternoon hours (12-4 PM)";
        avoidTime = '12:00 - 16:00';
      }
    }

    if (activity === 'outdoor' || activity === 'exercise') {
      if (temp >= 32) {
        recommendation = "Schedule for early morning (6-9 AM) or evening (5-7 PM)";
        bestTime = '06:00';
        avoidTime = '11:00 - 16:00';
        tips.push("Carry water and electrolytes");
        tips.push("Wear light, breathable clothing");
        riskLevel = 'Medium';
      }
      if (condition.includes('rain')) {
        tips.push("Consider indoor alternatives");
        riskLevel = 'High';
      }
    }

    if (activity === 'farming') {
      if (condition.includes('rain')) {
        recommendation = "Good day for indoor farm work. Avoid field operations";
        tips.push("Postpone pesticide and fertilizer application");
        tips.push("Check drainage systems");
      }
      if (temp >= 35) {
        recommendation = "Work in early morning (5-10 AM) or late evening";
        bestTime = '05:00';
        avoidTime = '11:00 - 17:00';
        tips.push("Hydrate workers frequently");
        tips.push("Provide shade breaks");
      }
    }

    if (activity === 'event') {
      if (condition.includes('rain')) {
        riskLevel = 'High';
        recommendation = "Have backup indoor venue ready";
        tips.push("Arrange canopy/tent coverage");
        tips.push("Prepare rain contingency plan");
      }
      if (temp >= 35) {
        tips.push("Arrange shade and cooling stations");
        tips.push("Provide water stations for guests");
      }
    }

    if (activity === 'delivery') {
      if (condition.includes('rain')) {
        tips.push("Protect packages from water damage");
        tips.push("Allow extra delivery time");
        riskLevel = 'Medium';
      }
      if (temp >= 35) {
        tips.push("Prioritize AC in vehicle during peak hours");
        tips.push("Take breaks in shade");
      }
    }

    // Risk-based adjustments
    if (risks.includes('avoid_rain') && condition.includes('rain')) {
      riskLevel = 'High';
      recommendation = "High rain probability. Consider postponing outdoor activities";
    }
    if (risks.includes('avoid_heat') && temp >= 32) {
      riskLevel = riskLevel === 'High' ? 'High' : 'Medium';
      recommendation = "High temperature expected. Best hours: before 10 AM or after 5 PM";
    }

    // Default tips if none generated
    if (tips.length === 0) {
      tips.push("Weather conditions are favorable");
      tips.push("Stay aware of any changes in forecast");
    }

    return {
      success: true,
      recommendation,
      bestTime,
      avoidTime,
      riskLevel,
      tips,
      activity,
      source: 'local'
    };
  }
};
