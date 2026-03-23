const express = require('express');
const { body, validationResult } = require('express-validator');
const { optionalAuth } = require('../middleware/auth');
const { normalizeUserProfile, buildContext } = require('../services/contextService');
const { getCurrentWeather } = require('../services/weatherService');
const { retrieveKnowledge } = require('../services/ragService');
const { generateInsight } = require('../services/llmService');
const { buildTriggerNotifications } = require('../services/notificationService');

const router = express.Router();

function parseUserProfile(req) {
  if (req.user) {
    return normalizeUserProfile(req.user);
  }

  const payload = req.body.userProfile || {};
  const lat = Number(payload?.location?.lat);
  const lon = Number(payload?.location?.lon);

  return {
    user_id: String(payload.user_id || ''),
    user_type: payload.user_type || 'general',
    location: {
      lat: Number.isFinite(lat) ? lat : null,
      lon: Number.isFinite(lon) ? lon : null,
      city: payload?.location?.city || '',
      state: payload?.location?.state || '',
    },
    profile: {
      vehicle: payload?.profile?.vehicle || null,
      distance: payload?.profile?.distance || null,
      weather_risks: Array.isArray(payload?.profile?.weather_risks) ? payload.profile.weather_risks : [],
      active_hours: payload?.profile?.active_hours || null,
    },
  };
}

// ===========================================
// @route   POST /api/intelligence/insight
// @desc    End-to-end personalized weather intelligence pipeline
// @access  Public (more personalization if authenticated)
// ===========================================
router.post('/insight', optionalAuth, [
  body('location.lat').optional().isFloat({ min: -90, max: 90 }),
  body('location.lon').optional().isFloat({ min: -180, max: 180 }),
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }

    const userProfile = parseUserProfile(req);
    const bodyLat = Number(req.body?.location?.lat);
    const bodyLon = Number(req.body?.location?.lon);

    const lat = Number.isFinite(bodyLat) ? bodyLat : userProfile.location.lat;
    const lon = Number.isFinite(bodyLon) ? bodyLon : userProfile.location.lon;

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return res.status(400).json({
        success: false,
        message: 'Location coordinates are required (provide token user location or location.lat/location.lon).',
      });
    }

    const weather = await getCurrentWeather(lat, lon);

    const context = buildContext({
      userProfile: {
        ...userProfile,
        location: {
          ...userProfile.location,
          lat,
          lon,
          city: userProfile.location.city || weather.city,
        },
      },
      weather,
    });

    const ragContext = retrieveKnowledge(context);
    const insight = await generateInsight({ context, ragContext });
    const notifications = buildTriggerNotifications({
      weather,
      userProfile: context.user_profile,
    });

    res.json({
      success: true,
      data: {
        insight: insight.text,
        llmSource: insight.source,
        weather,
        context,
        ragContext,
        notifications,
      },
    });
  } catch (error) {
    console.error('Intelligence pipeline error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate personalized intelligence',
      error: error.message,
    });
  }
});

// ===========================================
// @route   POST /api/intelligence/notifications/preview
// @desc    Evaluate event-based triggers without LLM call
// @access  Public
// ===========================================
router.post('/notifications/preview', [
  body('weather.temp').isFloat(),
  body('weather.rain_probability').isFloat({ min: 0, max: 1 }),
  body('weather.aqi').isFloat({ min: 0 }),
  body('weather.wind_speed').isFloat({ min: 0 }),
  body('userProfile.user_type').optional().isString(),
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ success: false, errors: errors.array() });
  }

  const notifications = buildTriggerNotifications({
    weather: req.body.weather,
    userProfile: req.body.userProfile || { user_type: 'general' },
  });

  return res.json({
    success: true,
    data: {
      notifications,
      count: notifications.length,
    },
  });
});

module.exports = router;
