const mongoose = require('mongoose');

const weatherReportSchema = new mongoose.Schema({
  // Reporter
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },

  // Location
  location: {
    city: { type: String, required: true },
    state: { type: String },
    coordinates: {
      latitude: { type: Number, required: true },
      longitude: { type: Number, required: true },
    },
  },

  // Report Type
  reportType: {
    type: String,
    enum: ['rain', 'no_rain', 'flood', 'heatwave', 'storm', 'fog', 'clear', 'other'],
    required: true,
  },

  // Details
  intensity: {
    type: String,
    enum: ['light', 'moderate', 'heavy', 'extreme'],
    default: 'moderate',
  },
  description: {
    type: String,
    maxlength: 500,
  },
  photo: {
    type: String, // URL to uploaded photo
  },

  // Validation
  isVerified: { type: Boolean, default: false },
  verifiedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  upvotes: { type: Number, default: 0 },
  downvotes: { type: Number, default: 0 },

  // Metadata
  createdAt: { type: Date, default: Date.now },
  expiresAt: { type: Date, default: () => new Date(Date.now() + 6 * 60 * 60 * 1000) }, // 6 hours
}, {
  timestamps: true,
});

// Index for geospatial queries
weatherReportSchema.index({ 'location.coordinates': '2dsphere' });

// Index for expiration
weatherReportSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

module.exports = mongoose.model('WeatherReport', weatherReportSchema);
