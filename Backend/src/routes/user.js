const express = require('express');
const { body, validationResult } = require('express-validator');
const User = require('../models/User');
const { protect } = require('../middleware/auth');

const router = express.Router();

// ===========================================
// @route   PUT /api/user/profile
// @desc    Update user profile
// @access  Private
// ===========================================
router.put('/profile', protect, [
  body('name').optional().trim().notEmpty(),
  body('email').optional().isEmail(),
  body('persona').optional().isIn(['general', 'driver', 'student', 'worker', 'farmer', 'elderly', 'homemaker', 'business']),
  body('language').optional().isIn(['en', 'hi', 'mr', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'pa']),
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }

    const allowedFields = ['name', 'email', 'persona', 'language', 'location'];
    const updates = {};

    allowedFields.forEach(field => {
      if (req.body[field] !== undefined) {
        updates[field] = req.body[field];
      }
    });

    const user = await User.findByIdAndUpdate(
      req.user._id,
      { $set: updates },
      { new: true, runValidators: true }
    );

    res.json({
      success: true,
      message: 'Profile updated',
      user: user.toPublicJSON(),
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Update failed',
      error: error.message,
    });
  }
});

// ===========================================
// @route   PUT /api/user/preferences
// @desc    Update user preferences
// @access  Private
// ===========================================
router.put('/preferences', protect, async (req, res) => {
  try {
    const { notifications, alerts, units, theme, liteMode } = req.body;

    const updates = {};
    if (notifications) updates['preferences.notifications'] = notifications;
    if (alerts) updates['preferences.alerts'] = alerts;
    if (units) updates['preferences.units'] = units;
    if (theme) updates['preferences.theme'] = theme;
    if (liteMode !== undefined) updates['preferences.liteMode'] = liteMode;

    const user = await User.findByIdAndUpdate(
      req.user._id,
      { $set: updates },
      { new: true }
    );

    res.json({
      success: true,
      message: 'Preferences updated',
      preferences: user.preferences,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Update failed',
      error: error.message,
    });
  }
});

// ===========================================
// @route   PUT /api/user/location
// @desc    Update user location
// @access  Private
// ===========================================
router.put('/location', protect, async (req, res) => {
  try {
    const { city, state, district, pincode, latitude, longitude } = req.body;

    const location = {
      city: city || '',
      state: state || '',
      district: district || '',
      pincode: pincode || '',
      coordinates: {
        latitude: latitude || null,
        longitude: longitude || null,
      },
    };

    const user = await User.findByIdAndUpdate(
      req.user._id,
      { $set: { location } },
      { new: true }
    );

    res.json({
      success: true,
      message: 'Location updated',
      location: user.location,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Update failed',
      error: error.message,
    });
  }
});

// ===========================================
// @route   POST /api/user/add-points
// @desc    Add points to user (for weather reports)
// @access  Private
// ===========================================
router.post('/add-points', protect, async (req, res) => {
  try {
    const { points, reason } = req.body;

    const user = await User.findByIdAndUpdate(
      req.user._id,
      { 
        $inc: { points: points || 5 },
      },
      { new: true }
    );

    res.json({
      success: true,
      message: `+${points || 5} points added!`,
      totalPoints: user.points,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to add points',
    });
  }
});

module.exports = router;
