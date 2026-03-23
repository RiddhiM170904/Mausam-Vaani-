const { supabase } = require("./supabase");
const { notificationConfig } = require("./config");

function ensureConfigured() {
  if (!notificationConfig.enabled || !supabase) {
    const err = new Error("Notification service is not configured");
    err.status = 503;
    throw err;
  }
}

async function upsertSubscription({ userId, subscription, deviceInfo }) {
  ensureConfigured();

  const endpoint = subscription?.endpoint;
  const p256dh = subscription?.keys?.p256dh;
  const auth = subscription?.keys?.auth;

  if (!userId || !endpoint || !p256dh || !auth) {
    const err = new Error("Invalid subscription payload");
    err.status = 400;
    throw err;
  }

  const { error } = await supabase
    .from("push_subscriptions")
    .upsert(
      {
        user_id: userId,
        endpoint,
        p256dh,
        auth,
        device_info: deviceInfo || "",
        is_active: true,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "endpoint" }
    );

  if (error) throw error;
  return true;
}

async function deactivateSubscriptionByEndpoint(endpoint) {
  ensureConfigured();

  if (!endpoint) {
    const err = new Error("endpoint required");
    err.status = 400;
    throw err;
  }

  const { error } = await supabase
    .from("push_subscriptions")
    .update({ is_active: false, updated_at: new Date().toISOString() })
    .eq("endpoint", endpoint);

  if (error) throw error;
  return true;
}

async function savePreferences({ userId, enabled, dailyCount, timezone }) {
  ensureConfigured();

  if (!userId) {
    const err = new Error("userId required");
    err.status = 400;
    throw err;
  }

  const safeDailyCount = Number(dailyCount) >= 9 ? 9 : 8;
  const payload = {
    user_id: userId,
    enabled: Boolean(enabled),
    daily_count: safeDailyCount,
    timezone: timezone || "Asia/Kolkata",
    updated_at: new Date().toISOString(),
  };

  const { error } = await supabase
    .from("notification_preferences")
    .upsert(payload, { onConflict: "user_id" });

  if (error) throw error;
  return true;
}

async function getActiveTargets() {
  ensureConfigured();

  const { data: subs, error: subErr } = await supabase
    .from("push_subscriptions")
    .select("user_id, endpoint, p256dh, auth, is_active")
    .eq("is_active", true);
  if (subErr) throw subErr;

  const { data: prefs, error: prefErr } = await supabase
    .from("notification_preferences")
    .select("user_id, enabled, daily_count, timezone");
  if (prefErr) throw prefErr;

  const prefMap = new Map((prefs || []).map((p) => [String(p.user_id), p]));

  return (subs || [])
    .map((s) => {
      const pref = prefMap.get(String(s.user_id)) || {
        enabled: true,
        daily_count: 9,
        timezone: "Asia/Kolkata",
      };
      return {
        userId: String(s.user_id),
        subscription: {
          endpoint: s.endpoint,
          keys: { p256dh: s.p256dh, auth: s.auth },
        },
        preference: pref,
      };
    })
    .filter((r) => r.preference.enabled !== false);
}

async function getDeliveryLogForDate({ userId, dateKey }) {
  ensureConfigured();

  const { data, error } = await supabase
    .from("notification_delivery_log")
    .select("slot_time")
    .eq("user_id", userId)
    .eq("slot_date", dateKey);
  if (error) throw error;

  const map = {};
  for (const row of data || []) {
    map[row.slot_time] = true;
  }
  return map;
}

async function insertDeliveryLog({ userId, slotDate, slotTime, source, title, body }) {
  ensureConfigured();

  const { error } = await supabase.from("notification_delivery_log").insert({
    user_id: userId,
    slot_date: slotDate,
    slot_time: slotTime,
    source,
    title,
    body,
  });
  if (error) throw error;
  return true;
}

module.exports = {
  upsertSubscription,
  deactivateSubscriptionByEndpoint,
  savePreferences,
  getActiveTargets,
  getDeliveryLogForDate,
  insertDeliveryLog,
};
