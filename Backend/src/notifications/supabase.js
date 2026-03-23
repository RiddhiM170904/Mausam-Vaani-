const { createClient } = require("@supabase/supabase-js");
const { notificationConfig } = require("./config");

const supabase = notificationConfig.enabled
  ? createClient(notificationConfig.supabaseUrl, notificationConfig.supabaseServiceRoleKey, {
      auth: { persistSession: false, autoRefreshToken: false },
    })
  : null;

module.exports = { supabase };
