function getDayPart(date = new Date()) {
  const h = date.getHours();
  if (h < 5) return 'night';
  if (h < 11) return 'morning';
  if (h < 16) return 'afternoon';
  if (h < 20) return 'evening';
  return 'night';
}

function inferLocationContext(city = '') {
  const value = String(city).toLowerCase();
  if (value.includes('industrial')) return 'industrial';
  if (value.includes('village') || value.includes('rural')) return 'rural';
  return 'urban';
}

function getIndiaSeason(monthIndex) {
  // India-oriented broad season buckets.
  if ([10, 11, 0, 1].includes(monthIndex)) return 'winter';
  if ([2, 3, 4, 5].includes(monthIndex)) return 'summer';
  if ([6, 7, 8].includes(monthIndex)) return 'monsoon';
  return 'post_monsoon';
}

export function normalizeUserProfile(userProfile = {}) {
  const lat = Number(userProfile?.location?.lat);
  const lon = Number(userProfile?.location?.lon);

  return {
    user_id: String(userProfile.user_id || ''),
    user_type: String(userProfile.user_type || 'general'),
    location: {
      lat: Number.isFinite(lat) ? lat : null,
      lon: Number.isFinite(lon) ? lon : null,
      city: String(userProfile?.location?.city || ''),
      state: String(userProfile?.location?.state || ''),
    },
    profile: {
      vehicle: userProfile?.profile?.vehicle || null,
      distance: userProfile?.profile?.distance || null,
      weather_risks: Array.isArray(userProfile?.profile?.weather_risks) ? userProfile.profile.weather_risks : [],
      active_hours: userProfile?.profile?.active_hours || null,
    },
  };
}

export function buildContext(userProfile, weather) {
  const now = new Date();
  const monthIndex = now.getMonth();

  return {
    generated_at: now.toISOString(),
    time: getDayPart(),
    local_hour: now.getHours(),
    month_name: now.toLocaleString('en-IN', { month: 'long' }),
    season_hint: getIndiaSeason(monthIndex),
    location_context: inferLocationContext(userProfile?.location?.city || weather?.city || ''),
    user_profile: userProfile,
    weather,
  };
}
