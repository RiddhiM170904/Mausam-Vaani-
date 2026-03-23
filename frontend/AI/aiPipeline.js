import { fetchWeatherByCoordinates } from './weatherApi.js';
import { normalizeUserProfile, buildContext } from './contextBuilder.js';
import { retrieveRagContext } from './ragEngine.js';
import { generateInsight } from './llmClient.js';
import { buildNotifications } from './notificationEngine.js';
import { fetchUserProfileFromSupabase } from './supabaseProfile.js';

export async function getPersonalizedInsight(payload = {}) {
  const requirements = payload?.requirements || payload?.user_requirements || '';
  const intent = String(payload?.intent || 'insight').toLowerCase();
  const plannerContext = payload?.planner_context || null;
  const dbProfile = await fetchUserProfileFromSupabase(payload?.user_id || payload?.userId);
  const mergedProfile = {
    ...(dbProfile || {}),
    ...(payload.user_profile || {}),
    location: {
      ...(dbProfile?.location || {}),
      ...(payload?.user_profile?.location || {}),
    },
    profile: {
      ...(dbProfile?.profile || {}),
      ...(payload?.user_profile?.profile || {}),
    },
  };

  const lat = Number(payload?.location?.lat ?? mergedProfile?.location?.lat);
  const lon = Number(payload?.location?.lon ?? mergedProfile?.location?.lon);

  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    throw new Error('location.lat and location.lon are required');
  }

  const userProfile = normalizeUserProfile(mergedProfile);
  const weather = await fetchWeatherByCoordinates(lat, lon);

  const context = buildContext(
    {
      ...userProfile,
      location: {
        ...userProfile.location,
        lat,
        lon,
        city: userProfile.location.city || weather.city,
      },
    },
    weather,
  );

  context.requirements = requirements;
  context.intent = intent;
  if (plannerContext && typeof plannerContext === 'object') {
    context.planner_context = plannerContext;
  }

  const ragContext = retrieveRagContext(context);
  const insight = await generateInsight(context, ragContext);
  const notifications = buildNotifications(weather, context.user_profile);

  return {
    success: true,
    data: {
      insight: insight.text,
      llm_source: insight.source,
      weather,
      context,
      rag_context: ragContext,
      notifications,
      source_profile: dbProfile ? 'supabase+payload' : 'payload_only',
    },
  };
}

export function previewNotifications(payload = {}) {
  const weather = payload.weather || {};
  const userProfile = payload.user_profile || {};
  const notifications = buildNotifications(weather, userProfile);

  return {
    success: true,
    data: {
      notifications,
      count: notifications.length,
    },
  };
}
