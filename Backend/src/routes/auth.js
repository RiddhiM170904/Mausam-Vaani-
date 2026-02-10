const express = require('express');
const { body, validationResult } = require('express-validator');
const User = require('../models/User');
const { protect } = require('../middleware/auth');

const router = express.Router();

// ===========================================
// @route   POST /api/auth/register
// @desc    Register new user with comprehensive onboarding
// @access  Public
// ===========================================
router.post('/register', [
  body('name').trim().notEmpty().withMessage('Name is required'),
  body('phone').matches(/^[6-9]\d{9}$/).withMessage('Valid Indian phone number required'),
  body('password').isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),
  body('persona').optional().isIn(['general', 'driver', 'worker', 'office_employee', 'farmer', 'delivery', 'senior_citizen', 'student', 'business_owner', 'other']),
  body('language').optional().isIn(['en', 'hi', 'mr', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'pa']),
  body('weatherRisks').optional().isArray(),
  body('activeHours.startTime').optional().matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).withMessage('Invalid start time format'),
  body('activeHours.endTime').optional().matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).withMessage('Invalid end time format'),
  body('notificationPreferences').optional().isIn(['severe_only', 'all_updates', 'daily_summary', 'none']),
], async (req, res) => {
  try {
    // Validation
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array(),
      });
    }

    const { 
      name, 
      phone, 
      password, 
      email, 
      persona, 
      otherPersonaText,
      language, 
      weatherRisks,
      activeHours,
      locations,
      notificationPreferences 
    } = req.body;

    // Check if user exists
    const existingUser = await User.findOne({ phone });
    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'Phone number already registered',
      });
    }

    // Create user with comprehensive data
    const userData = {
      name,
      phone,
      password,
      email,
      persona: persona || 'general',
      language: language || 'en',
      onboardingCompleted: true,
      onboardingStep: 6
    };

    // Add optional fields
    if (otherPersonaText && persona === 'other') {
      userData.otherPersonaText = otherPersonaText;
    }

    if (weatherRisks && Array.isArray(weatherRisks)) {
      userData.weatherRisks = weatherRisks;
    }

    if (activeHours) {
      userData.activeHours = {
        startTime: activeHours.startTime || '09:00',
        endTime: activeHours.endTime || '18:00',
        enabled: activeHours.enabled !== false
      };
    }

    if (locations && Array.isArray(locations) && locations.length > 0) {
      userData.locations = locations.map((location, index) => ({
        ...location,
        isPrimary: index === 0 // First location is primary
      }));
    }

    if (notificationPreferences) {
      userData.notificationPreferences = notificationPreferences;
    }

    // Create user
    const user = await User.create(userData);

    // Generate token
    const token = user.generateToken();

    res.status(201).json({
      success: true,
      message: 'Registration successful',
      token,
      user: user.toPublicJSON(),
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({
      success: false,
      message: 'Registration failed',
      error: error.message,
    });
  }
});

// ===========================================
// @route   POST /api/auth/login
// @desc    Login user
// @access  Public
// ===========================================
router.post('/login', [
  body('phone').matches(/^[6-9]\d{9}$/).withMessage('Valid phone number required'),
  body('password').notEmpty().withMessage('Password is required'),
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array(),
      });
    }

    const { phone, password } = req.body;

    // Find user
    const user = await User.findOne({ phone }).select('+password');
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials',
      });
    }

    // Check password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials',
      });
    }

    // Update last login
    user.lastLogin = new Date();
    await user.save();

    // Generate token
    const token = user.generateToken();

    res.json({
      success: true,
      message: 'Login successful',
      token,
      user: user.toPublicJSON(),
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({
      success: false,
      message: 'Login failed',
      error: error.message,
    });
  }
});

// ===========================================
// @route   GET /api/auth/me
// @desc    Get current user
// @access  Private
// ===========================================
router.get('/me', protect, async (req, res) => {
  res.json({
    success: true,
    user: req.user.toPublicJSON(),
  });
});

// ===========================================
// @route   POST /api/auth/logout
// @desc    Logout user (client-side token removal)
// @access  Private
// ===========================================
router.post('/logout', protect, (req, res) => {
  res.json({
    success: true,
    message: 'Logged out successfully',
  });
});

// ===========================================
// @route   PUT /api/auth/update-profile
// @desc    Update user profile/onboarding data
// @access  Private
// ===========================================
router.put('/update-profile', protect, [
  body('persona').optional().isIn(['general', 'driver', 'worker', 'office_employee', 'farmer', 'delivery', 'senior_citizen', 'student', 'business_owner', 'other']),
  body('language').optional().isIn(['en', 'hi', 'mr', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'pa']),
  body('weatherRisks').optional().isArray(),
  body('activeHours.startTime').optional().matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/),
  body('activeHours.endTime').optional().matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/),
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        errors: errors.array(),
      });
    }

    const user = req.user;
    const updates = req.body;

    // Update allowed fields
    Object.keys(updates).forEach(key => {
      if (['persona', 'otherPersonaText', 'language', 'weatherRisks', 'activeHours', 'locations', 'notificationPreferences', 'onboardingCompleted', 'onboardingStep'].includes(key)) {
        user[key] = updates[key];
      }
    });

    await user.save();

    res.json({
      success: true,
      message: 'Profile updated successfully',
      user: user.toPublicJSON(),
    });
  } catch (error) {
    console.error('Profile update error:', error);
    res.status(500).json({
      success: false,
      message: 'Profile update failed',
      error: error.message,
    });
  }
});

module.exports = router;
