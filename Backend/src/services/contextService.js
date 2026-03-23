function getDayPart(date = new Date()) {
  const hour = date.getHours();
  if (hour < 5) return 'night';
  if (hour < 11) return 'morning';
  if (hour < 16) return 'afternoon';
  if (hour < 20) return 'evening';
  return 'night';
}

function inferLocationContext(location = {}) {
  const city = (location.city || '').toLowerCase();
  const district = (location.district || '').toLowerCase();

  if ([city, district].some((v) => v.includes('industrial'))) {
    return 'industrial';
  }

  if ([city, district].some((v) => v.includes('village') || v.includes('rural'))) {
    return 'rural';
  }

  return 'urban';
}

function normalizeUserProfile(user = {}) {
  const lat = user.location?.coordinates?.latitude;
  const lon = user.location?.coordinates?.longitude;

  return {
    user_id: String(user._id || user.id || ''),
    user_type: user.persona || 'general',
    location: {
      lat: typeof lat === 'number' ? lat : null,
      lon: typeof lon === 'number' ? lon : null,
      city: user.location?.city || '',
      state: user.location?.state || '',
    },
    profile: {
      vehicle: user.profile?.vehicle || null,
      distance: user.profile?.distance || null,
      weather_risks: Array.isArray(user.weatherRisks) ? user.weatherRisks : [],
      active_hours: user.activeHours || null,
    },
  };
}

function buildContext({ userProfile, weather, now = new Date() }) {
  return {
    generated_at: now.toISOString(),
    time: getDayPart(now),
    location_context: inferLocationContext(userProfile.location),
    user_profile: userProfile,
    weather,
  };
}

module.exports = {
  normalizeUserProfile,
  buildContext,
};
