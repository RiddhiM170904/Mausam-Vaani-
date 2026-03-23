const ENV = process.env;

function readNumber(value, fallback) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

const notificationConfig = {
  enabled: Boolean(
    ENV.SUPABASE_URL &&
      ENV.SUPABASE_SERVICE_ROLE_KEY &&
      ENV.VAPID_SUBJECT &&
      ENV.VAPID_PUBLIC_KEY &&
      ENV.VAPID_PRIVATE_KEY
  ),
  nodeEnv: ENV.NODE_ENV || "development",
  timezone: ENV.NOTIFICATION_TIMEZONE || "Asia/Kolkata",
  supabaseUrl: ENV.SUPABASE_URL || "",
  supabaseServiceRoleKey: ENV.SUPABASE_SERVICE_ROLE_KEY || "",
  vapidSubject: ENV.VAPID_SUBJECT || "",
  vapidPublicKey: ENV.VAPID_PUBLIC_KEY || "",
  vapidPrivateKey: ENV.VAPID_PRIVATE_KEY || "",
  schedulerCron: ENV.NOTIFICATION_CRON || "*/5 * * * *",
  pushRetryLimit: readNumber(ENV.NOTIFICATION_PUSH_RETRY_LIMIT, 0),
};

module.exports = { notificationConfig };
