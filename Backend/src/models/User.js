const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const userSchema = new mongoose.Schema({
  // Basic Info
  name: {
    type: String,
    required: [true, 'Please provide your name'],
    trim: true,
    maxlength: [50, 'Name cannot exceed 50 characters'],
  },
  phone: {
    type: String,
    required: [true, 'Please provide your phone number'],
    unique: true,
    match: [/^[6-9]\d{9}$/, 'Please provide a valid Indian phone number'],
  },
  email: {
    type: String,
    unique: true,
    sparse: true,
    lowercase: true,
    match: [/^\S+@\S+\.\S+$/, 'Please provide a valid email'],
  },
  password: {
    type: String,
    required: [true, 'Please provide a password'],
    minlength: [6, 'Password must be at least 6 characters'],
    select: false, // Don't return password by default
  },

  // Location
  location: {
    city: { type: String, default: '' },
    state: { type: String, default: '' },
    district: { type: String, default: '' },
    pincode: { type: String, default: '' },
    coordinates: {
      latitude: { type: Number, default: null },
      longitude: { type: Number, default: null },
    },
  },

  // Personalization
  persona: {
    type: String,
    enum: ['general', 'driver', 'worker', 'office_employee', 'farmer', 'delivery', 'senior_citizen', 'student', 'business_owner', 'other'],
    default: 'general',
  },
  otherPersonaText: {
    type: String,
    trim: true,
    maxlength: [100, 'Persona description cannot exceed 100 characters'],
  },
  language: {
    type: String,
    enum: ['en', 'hi', 'mr', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'pa'],
    default: 'en',
  },

  // Weather Interests/Risks
  weatherRisks: [{
    type: String,
    enum: ['heavy_rain', 'flood', 'heatwave', 'strong_winds', 'fog', 'cold_waves', 'air_quality', 'storms', 'all']
  }],

  // Daily Schedule
  activeHours: {
    startTime: {
      type: String, // Format: "HH:MM"
      default: '09:00'
    },
    endTime: {
      type: String, // Format: "HH:MM"
      default: '18:00'
    },
    enabled: {
      type: Boolean,
      default: true
    }
  },

  // Multiple Locations Support
  locations: [{
    name: {
      type: String,
      required: true
    },
    type: {
      type: String,
      enum: ['home', 'work', 'current', 'other'],
      default: 'current'
    },
    city: String,
    state: String,
    district: String,
    pincode: String,
    coordinates: {
      latitude: Number,
      longitude: Number
    },
    isPrimary: {
      type: Boolean,
      default: false
    }
  }],

  // Current Active Location
  currentLocationIndex: {
    type: Number,
    default: 0
  },

  // Notification Preferences
  notificationPreferences: {
    type: String,
    enum: ['severe_only', 'all_updates', 'daily_summary', 'none'],
    default: 'severe_only'
  },

  // Other Preferences
  preferences: {
    notifications: {
      push: { type: Boolean, default: true },
      sms: { type: Boolean, default: false },
      email: { type: Boolean, default: false },
      whatsapp: { type: Boolean, default: true },
    },
    alerts: {
      rain: { type: Boolean, default: true },
      heatwave: { type: Boolean, default: true },
      aqi: { type: Boolean, default: true },
      storm: { type: Boolean, default: true },
    },
    units: {
      temperature: { type: String, enum: ['celsius', 'fahrenheit'], default: 'celsius' },
    },
    theme: { type: String, enum: ['dark', 'light', 'auto'], default: 'dark' },
    liteMode: { type: Boolean, default: false },
  },

  // Onboarding Status
  onboardingCompleted: {
    type: Boolean,
    default: false
  },
  onboardingStep: {
    type: Number,
    default: 0
  },

  // Gamification
  points: { type: Number, default: 0 },
  weatherReports: { type: Number, default: 0 },
  badges: [{ type: String }],

  // Metadata
  isVerified: { type: Boolean, default: false },
  lastLogin: { type: Date },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
}, {
  timestamps: true,
});

// Hash password before saving
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// Compare password method
userSchema.methods.comparePassword = async function(candidatePassword) {
  return await bcrypt.compare(candidatePassword, this.password);
};

// Generate JWT token
userSchema.methods.generateToken = function() {
  return jwt.sign(
    { id: this._id, phone: this.phone },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRE || '30d' }
  );
};

// Get public profile (without sensitive data)
userSchema.methods.toPublicJSON = function() {
  return {
    id: this._id,
    name: this.name,
    phone: this.phone,
    email: this.email,
    location: this.location,
    persona: this.persona,
    language: this.language,
    preferences: this.preferences,
    points: this.points,
    weatherReports: this.weatherReports,
    badges: this.badges,
    isVerified: this.isVerified,
    createdAt: this.createdAt,
  };
};

module.exports = mongoose.model('User', userSchema);
