import { isSupabaseConfigured, supabase } from '../src/services/supabaseClient';

function normalizeFromDb(row = {}) {
  const lat = Number(row?.latitude ?? row?.lat ?? row?.location?.lat);
  const lon = Number(row?.longitude ?? row?.lon ?? row?.location?.lon);

  return {
    user_id: String(row.id || ''),
    user_type: row.persona || 'general',
    location: {
      lat: Number.isFinite(lat) ? lat : null,
      lon: Number.isFinite(lon) ? lon : null,
      city: row.city || row?.location?.city || '',
      state: row.state || row?.location?.state || '',
    },
    profile: {
      vehicle: row.vehicle || row?.planner_profile?.answers?.vehicle || null,
      distance: row.distance || row?.planner_profile?.answers?.distance || null,
      weather_risks: Array.isArray(row.weather_risks)
        ? row.weather_risks
        : Array.isArray(row.weatherRisks)
          ? row.weatherRisks
          : [],
      active_hours: row.active_hours || null,
      planner_answers: row?.planner_profile?.answers || {},
    },
  };
}

export async function fetchUserProfileFromSupabase(userId) {
  if (!userId || !isSupabaseConfigured || !supabase) {
    return null;
  }

  const { data, error } = await supabase
    .from('users')
    .select('*')
    .eq('id', userId)
    .maybeSingle();

  if (error || !data) {
    return null;
  }

  return normalizeFromDb(data);
}
